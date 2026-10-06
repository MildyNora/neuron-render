// The renderer proper: one fitted scene, rendered from any camera, with its frame, lights and materials settable.
using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace NeuronRender
{
    [Serializable]
    public struct NeuronQuality
    {
        public float scale; public int aa; public int rate; public bool full; public int lobe;
        public static NeuronQuality Draft => new NeuronQuality { scale = 0.5f, aa = 2, rate = 4, lobe = 4 };
        public static NeuronQuality Fast => new NeuronQuality { scale = 1f, aa = 2, rate = 4, lobe = 4 };
        public static NeuronQuality Balanced => new NeuronQuality { scale = 1f, aa = 2, rate = 2, lobe = 8 };
        public static NeuronQuality High => new NeuronQuality { scale = 1f, aa = 4, rate = 1, lobe = 16 };
        public static NeuronQuality Ultra => new NeuronQuality { scale = 1f, aa = 2, rate = 1, full = true, lobe = 16 };
    }

    /// A camera in the scene's own (Blender, Z-up, right-handed) coordinates. Pixel y grows downwards.
    public struct NeuronCamera
    {
        public Vector3 origin, right, up, fwd;
        public float fx, fy, cx, cy, near;
        public int width, height;

        public static Vector3 ToBlender(Vector3 u) => new Vector3(u.x, u.z, u.y);   // Unity Y-up left-handed <-> Blender Z-up right-handed
        public static Vector3 ToUnity(Vector3 b) => new Vector3(b.x, b.z, b.y);

        public static NeuronCamera FromUnity(Camera cam, int width, int height)
        {
            var t = cam.transform;
            float fy = 0.5f * height / Mathf.Tan(0.5f * cam.fieldOfView * Mathf.Deg2Rad);
            return new NeuronCamera
            {
                origin = ToBlender(t.position), right = ToBlender(t.right), up = ToBlender(t.up), fwd = ToBlender(t.forward),
                fx = fy, fy = fy, cx = 0.5f * width, cy = 0.5f * height, near = Mathf.Max(cam.nearClipPlane, 1e-4f), width = width, height = height,
            };
        }

        /// From a Blender camera (4x4 camera-to-world, lens / sensor as in pack.json), at the pack's resolution scaled.
        public static NeuronCamera FromBlender(float[,] m, float lens, float sensorWidth, float sensorHeight, string sensorFit,
                                               float shiftX, float shiftY, float near, int width, int height)
        {
            var right = new Vector3(m[0, 0], m[1, 0], m[2, 0]).normalized;
            var up = new Vector3(m[0, 1], m[1, 1], m[2, 1]).normalized;
            var back = new Vector3(m[0, 2], m[1, 2], m[2, 2]).normalized;
            bool horizontal = sensorFit == "HORIZONTAL" || (sensorFit != "VERTICAL" && width >= height);
            float sensor = sensorFit == "VERTICAL" ? sensorHeight : sensorWidth;
            float viewfac = horizontal ? width : height;
            float f = lens / sensor * viewfac;
            return new NeuronCamera
            {
                origin = new Vector3(m[0, 3], m[1, 3], m[2, 3]), right = right, up = up, fwd = -back,
                fx = f, fy = f, cx = 0.5f * width - shiftX * viewfac, cy = 0.5f * height + shiftY * viewfac, near = near, width = width, height = height,
            };
        }
    }

    public sealed class NeuronScene : IDisposable
    {
        public readonly NeuronPack Pack;
        public NeuronQuality Quality = NeuronQuality.Balanced;

        // adjustable state
        public readonly Dictionary<string, Vector3> LightColor = new Dictionary<string, Vector3>();
        public readonly Dictionary<string, float> LightEnergy = new Dictionary<string, float>();     // lamps: energy; world: strength
        public readonly Dictionary<string, Vector3> MaterialColor = new Dictionary<string, Vector3>();
        public readonly Dictionary<string, float> MaterialRoughness = new Dictionary<string, float>();
        int frameIndex = -1, builtFrame = -2;
        bool materialsDirty = true;

        readonly ComputeShader shade, resolve, composite, frame;
        readonly Material idMaterial;
        readonly int kEncode, kGemm, kClear, kGroups, kArgs, kComposite, kBuild, kRefit;
        ComputeBuffer tri, bvh, materials, lights, cfg, icfg, vt, cap, subw, rcfg, counters, args, gtri, gdir, pxref, egroup, ew, pxmiss, feat, rad, hid0, hid1;
        ComputeBuffer objKind, affine, affineN, deformOffset, deformFirst, deformVerts, deformNormals;
        RenderTexture idTex, outTex;
        int capW, capH, capS, maxGroups;
        const int Chunk = 1 << 18;
        float[] materialTable;
        int[] capFlags, icfgNow;
        public int LastGroupCount { get; private set; }

        public NeuronScene(NeuronPack pack)
        {
            Pack = pack;
            shade = Resources.Load<ComputeShader>("NeuronShade");
            resolve = Resources.Load<ComputeShader>("NeuronResolve");
            composite = Resources.Load<ComputeShader>("NeuronComposite");
            frame = Resources.Load<ComputeShader>("NeuronFrame");
            idMaterial = new Material(Shader.Find("Hidden/NeuronRender/Id"));
            kEncode = shade.FindKernel("Encode"); kGemm = shade.FindKernel("Gemm");
            kClear = resolve.FindKernel("Clear"); kGroups = resolve.FindKernel("Groups"); kArgs = resolve.FindKernel("Args");
            kComposite = composite.FindKernel("Composite");
            kBuild = frame.FindKernel("BuildFrame"); kRefit = frame.FindKernel("RefitBvh");

            int T = pack.NumTris;
            tri = new ComputeBuffer((T + 1) * 24, 4);
            bvh = new ComputeBuffer(pack.BvhNodes * 8, 4); bvh.SetData(pack.BvhRef);
            materials = new ComputeBuffer(pack.MaterialsBase.Length, 4);
            lights = new ComputeBuffer(Math.Max(pack.LightRows, 1) * 20, 4);
            cfg = new ComputeBuffer(pack.Cfg.Length, 4); cfg.SetData(pack.Cfg);
            icfgNow = (int[])pack.Icfg.Clone();
            icfg = new ComputeBuffer(icfgNow.Length, 4); icfg.SetData(icfgNow);
            vt = new ComputeBuffer(1 + Math.Max(pack.DG, 0) + 1, 4);
            cap = new ComputeBuffer(T + 1, 4);
            subw = new ComputeBuffer(16, 4);
            rcfg = new ComputeBuffer(11 + 3 * pack.G, 4);
            counters = new ComputeBuffer(2, 4);
            args = new ComputeBuffer(3, 4, ComputeBufferType.IndirectArguments);
            feat = new ComputeBuffer(Chunk * ((pack.DIN + 1) / 2), 4);
            hid0 = new ComputeBuffer(Chunk * pack.MlpDims[1] / 2, 4); hid1 = new ComputeBuffer(Chunk * pack.MlpDims[1] / 2, 4);
            objKind = new ComputeBuffer(Math.Max(pack.ObjectCount, 1), 4); objKind.SetData(pack.ObjKind);
            affine = new ComputeBuffer(Math.Max(pack.ObjectCount, 1) * 12, 4);
            affineN = new ComputeBuffer(Math.Max(pack.ObjectCount, 1) * 9, 4);
            deformOffset = new ComputeBuffer(Math.Max(pack.ObjectCount, 1), 4);
            deformFirst = new ComputeBuffer(Math.Max(pack.ObjectCount, 1), 4); deformFirst.SetData(pack.DeformFirst);
            int deformRows = 0; var offs = new int[Math.Max(pack.ObjectCount, 1)];
            for (int k = 0; k < pack.ObjectCount; k++) { offs[k] = pack.ObjKind[k] == 2 ? deformRows : -1; if (pack.ObjKind[k] == 2) deformRows += pack.DeformCount[k]; }
            deformOffset.SetData(offs);
            deformVerts = new ComputeBuffer(Math.Max(deformRows, 1) * 9, 4);
            deformNormals = new ComputeBuffer(Math.Max(deformRows, 1) * 9, 4);
            Frame = 0;
        }

        public int FrameCount => Pack.Frames.Count;
        public int Frame { get => frameIndex; set => frameIndex = Mathf.Clamp(value, 0, Pack.Frames.Count - 1); }
        public void MarkMaterialsDirty() => materialsDirty = true;

        // ---- state -------------------------------------------------------------------------------------
        float[] MaterialTable()
        {
            var t = (float[])Pack.MaterialsBase.Clone();
            var mats = Pack.Materials;
            for (int i = 0; i < mats.Count; i++)
            {
                string name = (string)mats[i]["name"];
                if (MaterialRoughness.TryGetValue(name, out var r)) t[i * 12] = r;
                if (MaterialColor.TryGetValue(name, out var c)) { t[i * 12 + 8] = c.x; t[i * 12 + 9] = c.y; t[i * 12 + 10] = c.z; }
            }
            return t;
        }

        float[] GlobalState(float[] table)
        {
            var idx = Pack.Varied; var g = new float[idx.Length * 4];
            for (int j = 0; j < idx.Length; j++)
            {
                int i = idx[j];
                g[4 * j] = Mathf.Sqrt(Mathf.Max(table[i * 12 + 8], 0)); g[4 * j + 1] = Mathf.Sqrt(Mathf.Max(table[i * 12 + 9], 0));
                g[4 * j + 2] = Mathf.Sqrt(Mathf.Max(table[i * 12 + 10], 0)); g[4 * j + 3] = table[i * 12];
            }
            return g;
        }

        int[] CapFlags(float[] table)
        {
            int T = Pack.NumTris; var flags = new int[T + 1];
            var mats = Pack.Materials;
            for (int i = 0; i < T; i++)
            {
                int m = Pack.TriMaterial[i];
                float rough = table[m * 12], metal = (float)mats[m]["metallic"], trans = (float)mats[m]["transmission"];
                bool f = Pack.TriTextured[i] != 0 || (Pack.TriSmooth[i] != 0 && rough <= 0.3f);
                if (Pack.Parametric && rough <= 0.15f && Mathf.Max(metal, trans) >= 0.5f) f = true;
                flags[i] = f ? 1 : 0;
            }
            return flags;
        }

        float[] GroupWeights()
        {
            if (!Pack.Parametric) return new float[] { 1f, 1f, 1f };   // one group: the whole picture; lights are not adjustable
            var groups = Pack.Groups; var w = new float[3 * groups.Count];
            for (int g = 0; g < groups.Count; g++)
            {
                string name = (string)groups[g]["name"], kind = (string)groups[g]["kind"];
                if (kind == "light")
                {
                    float e0 = (float)groups[g]["energy"];
                    var c0 = groups[g]["color"].ToObject<float[]>();
                    float k = LightEnergy.TryGetValue(name, out var e) ? e / Mathf.Max(e0, 1e-12f) : 1f;
                    var c = LightColor.TryGetValue(name, out var cc) ? cc : new Vector3(c0[0], c0[1], c0[2]);
                    w[3 * g] = c.x * k; w[3 * g + 1] = c.y * k; w[3 * g + 2] = c.z * k;
                }
                else
                {
                    float s0 = groups[g]["strength"] != null ? (float)groups[g]["strength"] : 1f;
                    float k = LightEnergy.TryGetValue(name, out var s) ? s / Mathf.Max(s0, 1e-12f) : 1f;
                    w[3 * g] = w[3 * g + 1] = w[3 * g + 2] = k;
                }
            }
            return w;
        }

        void UpdateMaterials()
        {
            materialTable = MaterialTable();
            materials.SetData(materialTable);
            capFlags = CapFlags(materialTable);
            cap.SetData(capFlags);
            materialsDirty = false;
            UpdateVt();
        }

        void UpdateVt()
        {
            int n = Pack.Frames.Count;
            float tau = n < 2 ? 0f : frameIndex / (float)(n - 1);
            var g = Pack.Parametric ? GlobalState(materialTable) : new float[0];
            var v = new float[Math.Max(1 + g.Length, 2)];
            v[0] = tau; Array.Copy(g, 0, v, 1, g.Length);
            vt.SetData(v);
        }

        void BuildFrameTables()
        {
            int fi = frameIndex, no = Pack.ObjectCount;
            var A = new float[Math.Max(no, 1) * 12]; var AN = new float[Math.Max(no, 1) * 9];
            int deformRows = 0;
            for (int k = 0; k < no; k++)
            {
                if (Pack.ObjKind[k] == 1)
                {
                    var all = Pack.Affines[k];
                    Array.Copy(all, fi * 12, A, k * 12, 12);
                    var m = new Matrix4x4();
                    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) m[r, c] = A[k * 12 + r * 3 + c];   // m[r,c] = A[r][c]
                    m[3, 3] = 1f;
                    var inv = m.inverse;
                    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) AN[k * 9 + r * 3 + c] = inv[c, r];   // inv(A).T
                }
                else if (Pack.ObjKind[k] == 2) deformRows += Pack.DeformCount[k];
            }
            affine.SetData(A); affineN.SetData(AN);
            if (deformRows > 0)
            {
                var dv = new float[deformRows * 9]; var dn = new float[deformRows * 9]; int off = 0;
                for (int k = 0; k < no; k++)
                {
                    if (Pack.ObjKind[k] != 2) continue;
                    int n = Pack.DeformCount[k];
                    Array.Copy(Pack.DeformVerts[k], fi * n * 9, dv, off * 9, n * 9);
                    Array.Copy(Pack.DeformNormals[k], fi * n * 9, dn, off * 9, n * 9);
                    off += n;
                }
                deformVerts.SetData(dv); deformNormals.SetData(dn);
            }
            frame.SetBuffer(kBuild, "_TriRef", Pack.TriRefBuf); frame.SetBuffer(kBuild, "_TriObject", Pack.TriObjectBuf);
            frame.SetBuffer(kBuild, "_ObjKind", objKind); frame.SetBuffer(kBuild, "_Affine", affine); frame.SetBuffer(kBuild, "_AffineN", affineN);
            frame.SetBuffer(kBuild, "_DeformOffset", deformOffset); frame.SetBuffer(kBuild, "_DeformFirst", deformFirst);
            frame.SetBuffer(kBuild, "_DeformVerts", deformVerts); frame.SetBuffer(kBuild, "_DeformNormals", deformNormals);
            frame.SetBuffer(kBuild, "_Tri", tri);
            frame.SetInt("_NumTris", Pack.NumTris);
            frame.Dispatch(kBuild, (Pack.NumTris + 1 + 63) / 64, 1, 1);
            bool moving = false; foreach (var k in Pack.ObjKind) moving |= k != 0;
            if (moving || builtFrame == -2)
            {
                frame.SetBuffer(kRefit, "_Tri", tri); frame.SetBuffer(kRefit, "_Bvh", bvh);
                frame.SetBuffer(kRefit, "_BvhOrder", Pack.BvhOrder); frame.SetBuffer(kRefit, "_BvhDepth", Pack.BvhDepthBuf);
                frame.SetInt("_NumNodes", Pack.BvhNodes);
                for (int level = Pack.BvhMaxDepth; level >= 0; level--)
                {
                    frame.SetInt("_Level", level);
                    frame.Dispatch(kRefit, (Pack.BvhNodes + 63) / 64, 1, 1);
                }
            }
            // this frame's lamps
            int rows = Pack.LightRows, src = Pack.LightsAll.Length > rows * 20 ? fi : 0;
            var lt = new float[Math.Max(rows, 1) * 20];
            if (rows > 0) Array.Copy(Pack.LightsAll, src * rows * 20, lt, 0, rows * 20);
            lights.SetData(lt);
            builtFrame = frameIndex;
        }

        // ---- a frame ----------------------------------------------------------------------------------
        void EnsureTargets(int W, int H, int S)
        {
            if (idTex != null && capW == W && capH == H && capS == S) return;
            idTex?.Release(); outTex?.Release();
            idTex = new RenderTexture(W * S, H * S, 24, RenderTextureFormat.RFloat) { filterMode = FilterMode.Point };
            idTex.Create();
            outTex = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat) { enableRandomWrite = true, filterMode = FilterMode.Bilinear };
            outTex.Create();
            int n = W * H * S * S;
            if (n != maxGroups)
            {
                foreach (var b in new[] { gtri, gdir, pxref, egroup, ew, pxmiss, rad }) b?.Release();
                gtri = new ComputeBuffer(n, 4); gdir = new ComputeBuffer(n * 3, 4); egroup = new ComputeBuffer(n, 4); ew = new ComputeBuffer(n, 4);
                pxref = new ComputeBuffer(W * H, 4); pxmiss = new ComputeBuffer(W * H, 4); rad = new ComputeBuffer(n * 3 * Pack.G, 4);
                maxGroups = n;
            }
            capW = W; capH = H; capS = S;
        }

        static float[] PixelFilter(string kind, float width, int S)
        {
            var w = new float[S]; double sum = 0;
            for (int i = 0; i < S; i++)
            {
                double o = (i + 0.5) / S - 0.5, v;
                if (kind == "BOX") v = Math.Abs(o) <= 0.5 * width + 1e-9 ? 1.0 : 0.0;
                else if (kind == "GAUSSIAN") v = Math.Exp(-2.0 * Math.Pow(o * 6.0 / width, 2));
                else
                {
                    double x = o / width + 0.5;
                    v = 0.35875 - 0.48829 * Math.Cos(2 * Math.PI * x) + 0.14128 * Math.Cos(4 * Math.PI * x) - 0.01168 * Math.Cos(6 * Math.PI * x);
                    if (Math.Abs(o) >= 0.5 * width) v = 0.0;
                }
                w[i] = (float)v; sum += v;
            }
            if (sum <= 0) for (int i = 0; i < S; i++) w[i] = 1f;
            var w2 = new float[S * S]; double tot = 0;
            for (int i = 0; i < S; i++) for (int j = 0; j < S; j++) { w2[i * S + j] = w[i] * w[j]; tot += w2[i * S + j]; }
            for (int i = 0; i < S * S; i++) w2[i] = (float)(w2[i] / tot);
            return w2;
        }

        Matrix4x4 ViewProjection(NeuronCamera c, int W, int H, float fx, float fy, float cx, float cy)
        {
            // view: Blender world -> camera (x right, y up, -z forward); projection: our intrinsics, OpenGL clip conventions
            var V = Matrix4x4.identity;
            V.SetRow(0, new Vector4(c.right.x, c.right.y, c.right.z, -Vector3.Dot(c.right, c.origin)));
            V.SetRow(1, new Vector4(c.up.x, c.up.y, c.up.z, -Vector3.Dot(c.up, c.origin)));
            V.SetRow(2, new Vector4(-c.fwd.x, -c.fwd.y, -c.fwd.z, Vector3.Dot(c.fwd, c.origin)));
            float n = c.near, f = 1.0e5f;
            var P = Matrix4x4.zero;
            P.SetRow(0, new Vector4(2f * fx / W, 0, -(2f * cx / W - 1f), 0));
            P.SetRow(1, new Vector4(0, 2f * fy / H, -(1f - 2f * cy / H), 0));
            P.SetRow(2, new Vector4(0, 0, -(f + n) / (f - n), -2f * f * n / (f - n)));
            P.SetRow(3, new Vector4(0, 0, -1f, 0));
            return GL.GetGPUProjectionMatrix(P, true) * V;
        }

        /// Render into `target` (any RenderTexture; the picture is resized to it).
        public void Render(NeuronCamera cam, RenderTexture target)
        {
            if (ProfileStages) { StageMs.Clear(); stageClock = System.Diagnostics.Stopwatch.StartNew(); }
            var q = Quality;
            int W = Mathf.Max(1, Mathf.RoundToInt(cam.width * q.scale)), H = Mathf.Max(1, Mathf.RoundToInt(cam.height * q.scale));
            int S = Mathf.Clamp(q.aa, 1, 4), R = Mathf.Max(q.rate, 1);
            if (R * R * S * S > 64) R = Mathf.Max(1, (int)Mathf.Sqrt(64f / (S * S)));
            float sx = W / (float)cam.width, sy = H / (float)cam.height;
            float fx = cam.fx * sx, fy = cam.fy * sy, cx = cam.cx * sx, cy = cam.cy * sy;
            if (materialsDirty) UpdateMaterials();
            if (builtFrame != frameIndex) { BuildFrameTables(); UpdateVt(); }
            EnsureTargets(W, H, S);
            int lobe = Pack.FittedLobe > 0 ? Mathf.Clamp(q.lobe, 1, Pack.FittedLobe) : 0;
            if (icfgNow[19] != lobe) { icfgNow[19] = lobe; icfg.SetData(icfgNow); }

            // 1. visibility
            var cb = new CommandBuffer { name = "NeuronRender id" };
            cb.SetRenderTarget(idTex);
            cb.ClearRenderTarget(true, true, new Color(-1f, -1f, -1f, -1f));
            idMaterial.SetBuffer("_Tri", tri);
            idMaterial.SetMatrix("_NeuronVP", ViewProjection(cam, W * S, H * S, fx * S, fy * S, cx * S, cy * S));
            cb.DrawProcedural(Matrix4x4.identity, idMaterial, 0, MeshTopology.Triangles, 3 * Pack.NumTris, 1);
            Graphics.ExecuteCommandBuffer(cb);
            cb.Release();
            Sync("visibility");

            // 2. shading groups
            bool neuralSky = (string)Pack.Sky["mode"] == "neural";
            subw.SetData(PixelFilter((string)Pack.Meta["filter"]["type"], (float)Pack.Meta["filter"]["width"], S));
            resolve.SetBuffer(kClear, "_Counters", counters);
            resolve.Dispatch(kClear, 1, 1, 1);
            resolve.SetTexture(kGroups, "_IdTex", idTex);
            resolve.SetBuffer(kGroups, "_SubW", subw); resolve.SetBuffer(kGroups, "_Cap", cap); resolve.SetBuffer(kGroups, "_Counters", counters);
            resolve.SetBuffer(kGroups, "_GTri", gtri); resolve.SetBuffer(kGroups, "_GDir", gdir); resolve.SetBuffer(kGroups, "_PxRef", pxref);
            resolve.SetBuffer(kGroups, "_EGroup", egroup); resolve.SetBuffer(kGroups, "_EW", ew); resolve.SetBuffer(kGroups, "_PxMiss", pxmiss);
            resolve.SetInt("_W", W); resolve.SetInt("_H", H); resolve.SetInt("_S", S); resolve.SetInt("_R", R); resolve.SetInt("_Full", q.full ? 1 : 0);
            resolve.SetInt("_SkyId", neuralSky ? Pack.SkyId : -1); resolve.SetInt("_FlipId", FlipId ? 1 : 0);
            resolve.SetVector("_CamRight", cam.right); resolve.SetVector("_CamUp", cam.up); resolve.SetVector("_CamFwd", cam.fwd);
            resolve.SetFloat("_Fx", fx); resolve.SetFloat("_Fy", fy); resolve.SetFloat("_Cx", cx); resolve.SetFloat("_Cy", cy);
            resolve.Dispatch(kGroups, ((W + R - 1) / R + 7) / 8, ((H + R - 1) / R + 7) / 8, 1);
            Sync("groups");

            // 3. the network, in chunks of shading samples: features, then one tiled matrix product per layer
            BindShade(kEncode);
            shade.SetVector("_Origin", cam.origin);
            int dinp = Pack.DIN + (Pack.DIN & 1), width = Pack.MlpDims[1];
            shade.SetInt("_Din", Pack.DIN); shade.SetInt("_Dinp", dinp); shade.SetInt("_Width", width); shade.SetInt("_Nout", 3 * Pack.G);
            shade.SetInt("_Chunk", Chunk); resolve.SetInt("_Chunk", Chunk);
            resolve.SetBuffer(kArgs, "_Counters", counters); resolve.SetBuffer(kArgs, "_Args", args);
            shade.SetBuffer(kGemm, "_Counters", counters); shade.SetBuffer(kGemm, "_Rad", rad);
            for (int off = 0; off < maxGroups; off += Chunk)
            {
                resolve.SetInt("_Offset", off); resolve.SetInt("_Group", 64); resolve.SetInt("_ArgsX", 0); resolve.Dispatch(kArgs, 1, 1, 1);
                shade.SetInt("_Offset", off);
                shade.DispatchIndirect(kEncode, args);
                Sync("encode");
                resolve.SetInt("_Group", 64); resolve.SetInt("_ArgsX", (width + 63) / 64); resolve.Dispatch(kArgs, 1, 1, 1);
                Layer(feat, dinp, Pack.DIN, hid0, width, 0, true, false, off);
                Layer(hid0, width, width, hid1, width, 1, true, false, off);
                Layer(hid1, width, width, hid0, width, 2, true, false, off);
                resolve.SetInt("_ArgsX", 1); resolve.Dispatch(kArgs, 1, 1, 1);
                Layer(hid0, width, width, hid1, 3 * Pack.G, 3, false, true, off);
                Sync("mlp");
            }

            // 4. pixels
            var weights = GroupWeights();
            var rc = new float[11 + 3 * Pack.G];
            rc[0] = Pack.Eps; rc[1] = 1f / Pack.RadianceScale;
            if (!neuralSky && Pack.Sky["groups"] != null)
            {
                var sg = Pack.Sky["groups"].ToObject<float[][]>();
                for (int g = 0; g < sg.Length && g < Pack.G; g++) for (int c = 0; c < 3; c++) rc[2 + c] += sg[g][c] * weights[3 * g + c];
            }
            else if (!neuralSky && Pack.Sky["color"] != null) { var col = Pack.Sky["color"].ToObject<float[]>(); for (int c = 0; c < 3; c++) rc[2 + c] = col[c]; }
            rc[5] = Pack.LutSize; rc[6] = Pack.LutVmax; rc[7] = Pack.LutA; rc[8] = 0f; rc[9] = 0f; rc[10] = Pack.G;
            Array.Copy(weights, 0, rc, 11, weights.Length);
            rcfg.SetData(rc);
            composite.SetBuffer(kComposite, "_Rad", rad); composite.SetBuffer(kComposite, "_PxRef", pxref); composite.SetBuffer(kComposite, "_EGroup", egroup);
            composite.SetBuffer(kComposite, "_EW", ew); composite.SetBuffer(kComposite, "_PxMiss", pxmiss); composite.SetBuffer(kComposite, "_Lut", Pack.Lut);
            composite.SetBuffer(kComposite, "_Rcfg", rcfg); composite.SetTexture(kComposite, "_Out", outTex);
            composite.SetInt("_W", W); composite.SetInt("_H", H); composite.SetInt("_FlipOut", FlipOut ? 1 : 0);
            composite.Dispatch(kComposite, (W + 7) / 8, (H + 7) / 8, 1);
            Sync("composite");
            Graphics.Blit(outTex, target);
        }

        public bool FlipId = true, FlipOut = true;

        /// Profiling: when on, the CPU waits for the GPU after every stage and StageMs holds the split.
        public bool ProfileStages;
        public readonly Dictionary<string, double> StageMs = new Dictionary<string, double>();
        System.Diagnostics.Stopwatch stageClock;
        void Sync(string stage)
        {
            if (!ProfileStages) return;
            var tmp = new uint[2]; counters.GetData(tmp);   // a readback forces the GPU to finish
            if (stageClock == null) stageClock = System.Diagnostics.Stopwatch.StartNew();
            StageMs[stage] = StageMs.TryGetValue(stage, out var v) ? v + stageClock.Elapsed.TotalMilliseconds : stageClock.Elapsed.TotalMilliseconds;
            stageClock.Restart();
        }

        public int ReadGroupCount() { var c = new uint[2]; counters.GetData(c); LastGroupCount = (int)c[0]; return LastGroupCount; }

        /// Debug: the shading samples of the last frame (triangle, direction, log-radiance) and the features of the first chunk.
        public void Dump(string dir)
        {
            int n = ReadGroupCount();
            var t = new int[n]; gtri.GetData(t, 0, 0, n);
            var d = new float[3 * n]; gdir.GetData(d, 0, 0, 3 * n);
            var r = new float[3 * Pack.G * n]; rad.GetData(r, 0, 0, 3 * Pack.G * n);
            int m = Math.Min(n, Chunk), dinp = Pack.DIN + (Pack.DIN & 1);
            var f = new uint[m * dinp / 2]; feat.GetData(f, 0, 0, f.Length);
            System.IO.Directory.CreateDirectory(dir);
            Write(dir + "/gtri.bin", t); Write(dir + "/gdir.bin", d); Write(dir + "/rad.bin", r); Write(dir + "/feat.bin", f);
            System.IO.File.WriteAllText(dir + "/dump.json", "{\"n\": " + n + ", \"chunk\": " + m + ", \"dinp\": " + dinp + ", \"G\": " + Pack.G + "}");
        }
        static void Write(string path, Array a) { var b = new byte[Buffer.ByteLength(a)]; Buffer.BlockCopy(a, 0, b, 0, b.Length); System.IO.File.WriteAllBytes(path, b); }

        void Layer(ComputeBuffer x, int kstride, int k, ComputeBuffer y, int m, int layer, bool relu, bool outFloat, int rowOffset)
        {
            shade.SetBuffer(kGemm, "_X", x); shade.SetBuffer(kGemm, "_Y", y); shade.SetBuffer(kGemm, "_Wm", Pack.W[layer]); shade.SetBuffer(kGemm, "_Bm", Pack.B[layer]);
            shade.SetInt("_K", k); shade.SetInt("_Kstride", kstride); shade.SetInt("_M", m); shade.SetInt("_Mstride", m + (m & 1));
            shade.SetInt("_Relu", relu ? 1 : 0); shade.SetInt("_OutFloat", outFloat ? 1 : 0); shade.SetInt("_RowOffset", rowOffset);
            shade.DispatchIndirect(kGemm, args);
        }

        void BindShade(int k)
        {
            shade.SetBuffer(k, "_Tri", tri); shade.SetBuffer(k, "_TriRef", Pack.TrefBuf); shade.SetBuffer(k, "_Colors", Pack.Colors);
            shade.SetBuffer(k, "_Materials", materials); shade.SetBuffer(k, "_Lights", lights); shade.SetBuffer(k, "_Bvh", bvh);
            shade.SetBuffer(k, "_BvhOrder", Pack.BvhOrder); shade.SetBuffer(k, "_Ldec", Pack.Ldec); shade.SetBuffer(k, "_Env", Pack.Env);
            shade.SetBuffer(k, "_Cfg", cfg); shade.SetBuffer(k, "_Icfg", icfg); shade.SetBuffer(k, "_Lv", Pack.Lv); shade.SetBuffer(k, "_Llv", Pack.Llv);
            shade.SetBuffer(k, "_Table", Pack.Table); shade.SetBuffer(k, "_LTable", Pack.LTable); shade.SetBuffer(k, "_Emb", Pack.Emb); shade.SetBuffer(k, "_Vt", vt);
            shade.SetBuffer(k, "_Counters", counters); shade.SetBuffer(k, "_GTri", gtri); shade.SetBuffer(k, "_GDir", gdir);
            shade.SetBuffer(k, "_Feat", feat); shade.SetBuffer(k, "_Rad", rad);
            for (int i = 0; i < 4; i++) { shade.SetBuffer(k, "_W" + i, Pack.W[i]); shade.SetBuffer(k, "_B" + i, Pack.B[i]); }
        }

        public void Dispose()
        {
            foreach (var b in new[] { tri, bvh, materials, lights, cfg, icfg, vt, cap, subw, rcfg, counters, args, gtri, gdir, pxref, egroup, ew, pxmiss, feat, rad, hid0, hid1,
                                      objKind, affine, affineN, deformOffset, deformFirst, deformVerts, deformNormals }) b?.Release();
            idTex?.Release(); outTex?.Release();
        }
    }
}
