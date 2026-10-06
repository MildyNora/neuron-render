// Headless checks: do the shaders compile, and does a frame rendered here match the Python renderer?
//   Unity -batchmode -projectPath unity/NeuronRender -executeMethod NeuronRender.Editor.NeuronValidate.Run -quit \
//         -nrpack /path/to/stage.nrpack -nrframe 0 -nrout /tmp/unity_hero.png [-nrpreset high]
using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace NeuronRender.Editor
{
    public static class NeuronValidate
    {
        static string Arg(string name, string fallback = null)
        {
            var a = Environment.GetCommandLineArgs();
            for (int i = 0; i < a.Length - 1; i++) if (a[i] == name) return a[i + 1];
            return fallback;
        }

        [MenuItem("Neuron Render/Check shaders")]
        public static bool CheckShaders()
        {
            bool ok = true;
            foreach (var name in new[] { "NeuronShade", "NeuronResolve", "NeuronComposite", "NeuronFrame" })
            {
                var cs = Resources.Load<ComputeShader>(name);
                if (cs == null) { Debug.LogError("missing compute shader " + name); ok = false; continue; }
                foreach (var m in ShaderUtil.GetComputeShaderMessages(cs))
                    if (m.severity == UnityEditor.Rendering.ShaderCompilerMessageSeverity.Error) { Debug.LogError(name + ": " + m.message + " (line " + m.line + ")"); ok = false; }
            }
            var sh = Shader.Find("Hidden/NeuronRender/Id");
            if (sh == null || ShaderUtil.ShaderHasError(sh)) { Debug.LogError("id shader missing or failed to compile"); ok = false; }
            Debug.Log(ok ? "NR_SHADERS_OK" : "NR_SHADERS_FAILED");
            return ok;
        }

        public static void Run()
        {
            bool ok = CheckShaders();
            string packPath = Arg("-nrpack");
            if (!ok || packPath == null) { EditorApplication.Exit(ok ? 0 : 1); return; }
            int frame = int.Parse(Arg("-nrframe", "0"));
            string outPath = Arg("-nrout", "/tmp/neuron_unity.png");
            string presetName = Arg("-nrpreset", "high");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var pack = new NeuronPack(packPath);
            Debug.Log("pack loaded in " + sw.ElapsedMilliseconds + " ms");
            var scene = new NeuronScene(pack);
            scene.Quality = presetName switch { "draft" => NeuronQuality.Draft, "fast" => NeuronQuality.Fast, "balanced" => NeuronQuality.Balanced, "ultra" => NeuronQuality.Ultra, _ => NeuronQuality.High };
            scene.Frame = frame;
            var cams = pack.Meta["cameras"];
            var mj = ((Newtonsoft.Json.Linq.JArray)cams)[Math.Min(frame, ((Newtonsoft.Json.Linq.JArray)cams).Count - 1)]["matrix"];
            var m = new float[4, 4];
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) m[r, c] = (float)mj[r][c];
            var cam = NeuronCamera.FromBlender(m, (float)((Newtonsoft.Json.Linq.JArray)cams)[Math.Min(frame, ((Newtonsoft.Json.Linq.JArray)cams).Count - 1)]["lens"], (float)pack.Meta["sensor_width"], (float)pack.Meta["sensor_height"],
                                               (string)pack.Meta["sensor_fit"], (float)pack.Meta["shift_x"], (float)pack.Meta["shift_y"], (float)pack.Meta["clip_start"], pack.Width, pack.Height);
            if (Arg("-nredit") != null)
            {   // the same edit the Python side renders for the comparison
                scene.MaterialColor["Paint"] = new Vector3(0.1f, 0.3f, 0.8f); scene.MaterialRoughness["Paint"] = 0.15f;
                scene.LightEnergy["Sun"] = 1f; scene.LightColor["Sun"] = new Vector3(1f, 0.7f, 0.4f); scene.LightEnergy["World"] = 0.3f;
                scene.MarkMaterialsDirty();
            }
            var rt = new RenderTexture(pack.Width, pack.Height, 0, RenderTextureFormat.ARGBFloat);
            rt.Create();
            if (Arg("-nrcomponent") != null)
            {   // through the camera component instead: tests the Unity-camera -> scene-camera conversion as a user would hit it
                var go = new GameObject("cam"); var ucam = go.AddComponent<Camera>();
                var jm = ((Newtonsoft.Json.Linq.JArray)cams)[Math.Min(frame, ((Newtonsoft.Json.Linq.JArray)cams).Count - 1)]["matrix"];
                var right = new Vector3((float)jm[0][0], (float)jm[2][0], (float)jm[1][0]);
                var up = new Vector3((float)jm[0][1], (float)jm[2][1], (float)jm[1][1]);
                var back = new Vector3((float)jm[0][2], (float)jm[2][2], (float)jm[1][2]);
                go.transform.position = new Vector3((float)jm[0][3], (float)jm[2][3], (float)jm[1][3]);
                go.transform.rotation = Quaternion.LookRotation(-back, up);
                ucam.fieldOfView = 2f * Mathf.Atan(0.5f * pack.Height / cam.fy) * Mathf.Rad2Deg;
                ucam.nearClipPlane = cam.near; ucam.targetTexture = rt;
                var comp = go.AddComponent<NeuronRenderer>(); comp.packPath = packPath; comp.frame = frame; comp.preset = NeuronRenderer.Preset.High;
                ucam.Render(); ucam.Render();
                var tc = new Texture2D(pack.Width, pack.Height, TextureFormat.RGBAFloat, false);
                RenderTexture.active = rt; tc.ReadPixels(new Rect(0, 0, pack.Width, pack.Height), 0, 0); RenderTexture.active = null;
                var pc = tc.GetPixels(); var rawc = new float[pc.Length * 3];
                for (int y = 0; y < pack.Height; y++) for (int x = 0; x < pack.Width; x++) { var c = pc[(pack.Height - 1 - y) * pack.Width + x]; int i = y * pack.Width + x; rawc[3 * i] = c.r; rawc[3 * i + 1] = c.g; rawc[3 * i + 2] = c.b; }
                var bc = new byte[rawc.Length * 4]; Buffer.BlockCopy(rawc, 0, bc, 0, bc.Length);
                File.WriteAllBytes(Path.ChangeExtension(outPath, ".component.f32"), bc);
                Debug.Log("NR_COMPONENT_OK ms=" + comp.LastFrameMs.ToString("F1"));
                comp.Unload(); UnityEngine.Object.DestroyImmediate(go);
            }
            scene.Render(cam, rt);                      // first frame: shader warm-up
            sw.Restart();
            const int reps = 10;
            for (int i = 0; i < reps; i++) scene.Render(cam, rt);
            var tex = new Texture2D(pack.Width, pack.Height, TextureFormat.RGBAFloat, false);
            RenderTexture.active = rt; tex.ReadPixels(new Rect(0, 0, pack.Width, pack.Height), 0, 0); RenderTexture.active = null;
            double ms = sw.Elapsed.TotalMilliseconds / reps;
            int groups = scene.ReadGroupCount();
            scene.ProfileStages = true; scene.Render(cam, rt); scene.ProfileStages = false;
            foreach (var kv in scene.StageMs) Debug.Log("NR_STAGE " + kv.Key + " " + kv.Value.ToString("F1") + " ms");
            string dump = Arg("-nrdump"); if (dump != null) { scene.Render(cam, rt); scene.Dump(dump); }
            // ReadPixels gives rows bottom-up; write the image top-down as the pack defines it
            var px = tex.GetPixels(); var flipped = new Color[px.Length];
            for (int y = 0; y < pack.Height; y++) Array.Copy(px, y * pack.Width, flipped, (pack.Height - 1 - y) * pack.Width, pack.Width);
            tex.SetPixels(flipped);
            var raw = new float[px.Length * 3];
            for (int i = 0; i < px.Length; i++) { raw[3 * i] = flipped[i].r; raw[3 * i + 1] = flipped[i].g; raw[3 * i + 2] = flipped[i].b; }
            var bytes = new byte[raw.Length * 4]; Buffer.BlockCopy(raw, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.ChangeExtension(outPath, ".f32"), bytes);
            for (int i = 0; i < px.Length; i++) flipped[i] = new Color(Mathf.Clamp01(flipped[i].r), Mathf.Clamp01(flipped[i].g), Mathf.Clamp01(flipped[i].b), 1);
            tex.SetPixels(flipped); tex.Apply();
            File.WriteAllBytes(outPath, tex.EncodeToPNG());
            Debug.Log("NR_RENDER_OK frame=" + frame + " preset=" + presetName + " groups=" + groups + " ms=" + ms.ToString("F1") + " out=" + outPath);
            scene.Dispose(); pack.Dispose();
            EditorApplication.Exit(0);
        }
    }
}
