// A fitted scene exported by `neuron-render export`: pack.json plus raw arrays. Loaded into GPU buffers here.
using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace NeuronRender
{
    public sealed class NeuronPack : IDisposable
    {
        public readonly string Path;
        public readonly JObject Meta;
        public readonly bool Parametric;
        public readonly int Width, Height, NumTris, SkyId, G, NL, DIN, L, F, FT, LL, FL, KT, DG, FittedLobe;
        public readonly bool Follow;
        public readonly int[] Icfg;
        public readonly float[] Cfg;
        public readonly float Eps, RadianceScale, Dither;
        public readonly int LutSize; public readonly float LutVmax, LutA;
        public readonly List<int> Frames = new List<int>();
        public readonly int[] TriObject, TriMaterial; public readonly byte[] TriSmooth, TriTextured;
        public readonly float[] MaterialsBase;         // [M + 1, 12]
        public readonly float[] LightsAll;             // [F or 1, rows, 20]
        public readonly int LightRows;
        public readonly float[] BvhRef; public readonly int[] BvhDepth; public readonly int BvhNodes, BvhMaxDepth;
        public readonly float[] TriRef;
        public readonly int ObjectCount;
        public readonly int[] ObjKind;                  // 0 static, 1 rigid, 2 deform
        public readonly Dictionary<int, float[]> Affines = new Dictionary<int, float[]>();   // object -> [F, 12]
        public readonly Dictionary<int, float[]> DeformVerts = new Dictionary<int, float[]>(), DeformNormals = new Dictionary<int, float[]>();
        public readonly int[] DeformFirst, DeformCount;

        // GPU resident tables
        public ComputeBuffer TriRefBuf, TrefBuf, Colors, Table, LTable, Emb, Ldec, Env, Lut, Lv, Llv, BvhOrder, BvhDepthBuf, TriObjectBuf;
        public ComputeBuffer[] W = new ComputeBuffer[4], B = new ComputeBuffer[4];
        public int[] MlpDims;

        public NeuronPack(string path)
        {
            Path = path;
            Meta = JObject.Parse(File.ReadAllText(System.IO.Path.Combine(path, "pack.json")));
            Parametric = (bool)Meta["parametric"];
            Width = (int)Meta["width"]; Height = (int)Meta["height"];
            NumTris = (int)Meta["n_tris"]; SkyId = (int)Meta["sky_id"];
            var dims = Meta["dims"];
            G = (int)dims["G"]; NL = (int)dims["NL"]; DIN = (int)dims["DIN"]; L = (int)dims["L"]; F = (int)dims["F"]; FT = (int)dims["FT"];
            LL = (int)dims["LL"]; FL = (int)dims["FL"]; KT = (int)dims["KT"]; DG = (int)dims["DG"];
            Icfg = Meta["icfg"].ToObject<int[]>(); Cfg = Meta["cfg_f"].ToObject<float[]>();
            FittedLobe = (int)Meta["lobe"]; Follow = (bool)Meta["follow"];
            Eps = (float)Meta["eps"]; RadianceScale = (float)Meta["radiance_scale"]; Dither = (float)Meta["dither"];
            LutSize = (int)Meta["lut"]["size"]; LutVmax = (float)Meta["lut"]["vmax"]; LutA = (float)Meta["lut"]["a"];
            foreach (var f in Meta["frames"]) Frames.Add((int)f);

            TriRef = ReadFloat("tri_ref");
            TriObject = ReadInt("tri_object"); TriMaterial = ReadInt("tri_material");
            TriSmooth = ReadBytes("tri_smooth"); TriTextured = ReadBytes("tri_textured");
            MaterialsBase = ReadFloat("materials");
            LightsAll = ReadFloat("lights"); LightRows = Shape("lights")[1];
            BvhRef = ReadFloat("bvh"); BvhNodes = Shape("bvh")[0]; BvhDepth = ReadInt("bvh_depth");
            foreach (var d in BvhDepth) BvhMaxDepth = Math.Max(BvhMaxDepth, d);
            var objects = (JArray)Meta["objects"];
            ObjectCount = objects.Count;
            ObjKind = new int[ObjectCount]; DeformFirst = new int[ObjectCount]; DeformCount = new int[ObjectCount];
            for (int k = 0; k < ObjectCount; k++)
            {
                string motion = (string)objects[k]["motion"];
                if (motion == "rigid") { ObjKind[k] = 1; Affines[k] = ReadFloat((string)objects[k]["affine"]); }
                else if (motion == "deform")
                {
                    ObjKind[k] = 2;
                    DeformVerts[k] = ReadFloat((string)objects[k]["verts"]); DeformNormals[k] = ReadFloat((string)objects[k]["normals"]);
                    DeformFirst[k] = (int)objects[k]["first"]; DeformCount[k] = (int)objects[k]["count"];
                }
            }

            TriRefBuf = Upload(TriRef);
            TrefBuf = Upload(ReadFloat("tref"));
            Colors = Upload(ReadFloat("colors"));
            Table = UploadHalves("table"); LTable = UploadHalves("ltable"); Emb = UploadHalves("emb"); Lut = UploadHalves("lut");
            Ldec = Upload(Has("ldec") ? ReadFloat("ldec") : new float[4]);
            Lv = Upload(ReadInt("lv")); Llv = Upload(ReadInt("llv"));
            BvhOrder = Upload(ReadInt("bvh_order")); BvhDepthBuf = Upload(BvhDepth); TriObjectBuf = Upload(TriObject);
            if (Has("env"))
            {
                // stored as radiance; the kernels read log(min(radiance * scale, vmax) + eps)
                var e = ReadHalfAsFloat("env");
                for (int i = 0; i < e.Length; i++) e[i] = Mathf.Log(Mathf.Min(e[i] * RadianceScale, LutVmax) + Eps);
                Env = Upload(e);
            }
            else Env = Upload(new float[3]);
            MlpDims = new int[5];
            for (int i = 0; i < 4; i++)
            {
                var sh = Shape("mlp_w" + i);
                MlpDims[i] = sh[1]; MlpDims[i + 1] = sh[0];
                W[i] = UploadHalves("mlp_w" + i); B[i] = UploadHalves("mlp_b" + i);
            }
        }

        public string[] GroupNames { get { var a = (JArray)Meta["groups"]; var n = new string[a.Count]; for (int i = 0; i < n.Length; i++) n[i] = (string)a[i]["name"]; return n; } }
        public JArray Groups => (JArray)Meta["groups"];
        public JArray Materials => (JArray)Meta["materials"];
        public int[] Varied => Meta["varied"].ToObject<int[]>();
        public JObject Sky => (JObject)Meta["sky"];
        public int MaterialCount => Materials.Count;

        bool Has(string name) => Meta["arrays"][name] != null;
        int[] Shape(string name) => Meta["arrays"][name]["shape"].ToObject<int[]>();
        byte[] Raw(string name) => File.ReadAllBytes(System.IO.Path.Combine(Path, (string)Meta["arrays"][name]["file"]));
        float[] ReadFloat(string name) { var b = Raw(name); var a = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, a, 0, b.Length); return a; }
        int[] ReadInt(string name) { var b = Raw(name); var a = new int[b.Length / 4]; Buffer.BlockCopy(b, 0, a, 0, b.Length); return a; }
        byte[] ReadBytes(string name) => Raw(name);
        uint[] ReadHalvesPacked(string name) { var b = Raw(name); var a = new uint[(b.Length + 3) / 4]; Buffer.BlockCopy(b, 0, a, 0, b.Length); return a; }
        float[] ReadHalfAsFloat(string name)
        {
            var b = Raw(name); var a = new float[b.Length / 2];
            for (int i = 0; i < a.Length; i++) a[i] = Mathf.HalfToFloat(BitConverter.ToUInt16(b, 2 * i));
            return a;
        }
        static ComputeBuffer Upload(float[] a) { var cb = new ComputeBuffer(Math.Max(a.Length, 1), 4); cb.SetData(a); return cb; }
        static ComputeBuffer Upload(int[] a) { var cb = new ComputeBuffer(Math.Max(a.Length, 1), 4); cb.SetData(a); return cb; }
        ComputeBuffer UploadHalves(string name) { var a = ReadHalvesPacked(name); var cb = new ComputeBuffer(Math.Max(a.Length, 1), 4); cb.SetData(a); return cb; }

        public void Dispose()
        {
            foreach (var cb in new[] { TriRefBuf, TrefBuf, Colors, Table, LTable, Emb, Ldec, Env, Lut, Lv, Llv, BvhOrder, BvhDepthBuf, TriObjectBuf }) cb?.Release();
            for (int i = 0; i < 4; i++) { W[i]?.Release(); B[i]?.Release(); }
        }
    }
}
