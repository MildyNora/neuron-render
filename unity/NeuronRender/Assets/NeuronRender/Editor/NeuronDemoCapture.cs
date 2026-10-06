// Records the Unity runtime at work, headlessly: the animation through the camera component, a camera move,
// lights and materials changed live. Writes numbered PNGs; demo/make_unity_video.py captions and encodes them.
//   Unity -batchmode -projectPath unity/NeuronRender -executeMethod NeuronRender.Editor.NeuronDemoCapture.Run -quit -nrpack <pack> -nrout <dir>
using System;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEngine;

namespace NeuronRender.Editor
{
    public static class NeuronDemoCapture
    {
        static string Arg(string name, string fallback = null)
        {
            var a = Environment.GetCommandLineArgs();
            for (int i = 0; i < a.Length - 1; i++) if (a[i] == name) return a[i + 1];
            return fallback;
        }

        static void Pose(Transform t, JToken m)
        {
            var right = new Vector3((float)m[0][0], (float)m[2][0], (float)m[1][0]);
            var up = new Vector3((float)m[0][1], (float)m[2][1], (float)m[1][1]);
            var back = new Vector3((float)m[0][2], (float)m[2][2], (float)m[1][2]);
            t.position = new Vector3((float)m[0][3], (float)m[2][3], (float)m[1][3]);
            t.rotation = Quaternion.LookRotation(-back, up);
        }

        public static void Run()
        {
            string packPath = Arg("-nrpack"), outDir = Arg("-nrout", "/tmp/neuron_unity_demo");
            string presetName = Arg("-nrpreset", "balanced");
            Directory.CreateDirectory(outDir);
            var meta = JObject.Parse(File.ReadAllText(Path.Combine(packPath, "pack.json")));
            int W = (int)meta["width"], H = (int)meta["height"];
            var cams = (JArray)meta["cameras"];
            var go = new GameObject("cam"); var cam = go.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = Color.black; cam.nearClipPlane = 0.05f;
            var rt = new RenderTexture(W, H, 0, RenderTextureFormat.ARGBFloat); rt.Create(); cam.targetTexture = rt;
            var nr = go.AddComponent<NeuronRenderer>(); nr.packPath = packPath;
            nr.preset = presetName == "high" ? NeuronRenderer.Preset.High : presetName == "fast" ? NeuronRenderer.Preset.Fast : NeuronRenderer.Preset.Balanced;
            float lens = (float)cams[0]["lens"], sw = (float)meta["sensor_width"];
            float f = lens / sw * W;
            cam.fieldOfView = 2f * Mathf.Atan(0.5f * H / f) * Mathf.Rad2Deg;
            nr.Load();
            var tex = new Texture2D(W, H, TextureFormat.RGBAFloat, false);
            var log = new System.Text.StringBuilder();
            int n = 0, frames = cams.Count;
            var center = NeuronCamera.ToUnity(new Vector3((float)meta["cfg_f"][0], (float)meta["cfg_f"][1], (float)meta["cfg_f"][2]));
            var sun = nr.lights.Find(l => l.name == "Sun"); var world = nr.lights.Find(l => l.isWorld);
            var paint = nr.materials.Find(m => m.name == "Paint"); var brushed = nr.materials.Find(m => m.name == "Brushed");
            Color sun0 = sun != null ? sun.color : Color.white; float sunE = sun != null ? sun.energy : 1, worldS = world != null ? world.energy : 1;
            Color paint0 = paint != null ? paint.baseColor : Color.white; float paintR = paint != null ? paint.roughness : 0.3f;
            float brushedR = brushed != null ? brushed.roughness : 0.3f;
            void Shoot(string part)
            {
                var t0 = System.Diagnostics.Stopwatch.StartNew();
                cam.Render();
                RenderTexture.active = rt; tex.ReadPixels(new Rect(0, 0, W, H), 0, 0); RenderTexture.active = null;
                double ms = t0.Elapsed.TotalMilliseconds;
                var px = tex.GetPixels();   // bottom row first, which is also what EncodeToPNG expects
                for (int i = 0; i < px.Length; i++) px[i] = new Color(Mathf.Clamp01(px[i].r), Mathf.Clamp01(px[i].g), Mathf.Clamp01(px[i].b), 1);
                tex.SetPixels(px); tex.Apply();
                File.WriteAllBytes(Path.Combine(outDir, string.Format("f_{0:D4}.png", n)), tex.EncodeToPNG());
                log.AppendFormat("{0} {1} {2} {3:F1}\n", n, part, nr.frame, nr.LastFrameMs > 0 ? nr.LastFrameMs : ms);
                n++;
            }
            // 1. the animation through its own camera path
            for (int i = 0; i < frames; i++) { nr.frame = i; Pose(go.transform, cams[i]["matrix"]); Shoot("animation"); }
            // 2. a camera move of our own, frame held
            nr.frame = frames / 2; Pose(go.transform, cams[frames / 2]["matrix"]);
            var p0 = go.transform.position; var rel = p0 - center; float r0 = new Vector2(rel.x, rel.z).magnitude, a0 = Mathf.Atan2(rel.z, rel.x);
            for (int i = 0; i < 72; i++)
            {
                float a = a0 + Mathf.Sin(i / 72f * Mathf.PI * 2f) * 0.35f, h = rel.y + Mathf.Sin(i / 72f * Mathf.PI * 2f) * 0.4f;
                go.transform.position = center + new Vector3(Mathf.Cos(a) * r0, h, Mathf.Sin(a) * r0);
                go.transform.LookAt(center + Vector3.up * 0.6f);
                Shoot("camera");
            }
            Pose(go.transform, cams[frames / 2]["matrix"]);
            // 3. lights
            for (int i = 0; i < 72; i++)
            {
                float u = i / 72f;
                if (sun != null) { sun.color = Color.HSVToRGB(u, 0.6f * Mathf.Sin(u * Mathf.PI), 1f); sun.energy = sunE * (1f + 0.8f * Mathf.Sin(u * Mathf.PI * 2f)); }
                if (world != null) world.energy = worldS * (1f + 0.7f * Mathf.Sin(u * Mathf.PI * 2f + 1f));
                Shoot("lights");
            }
            if (sun != null) { sun.color = sun0; sun.energy = sunE; } if (world != null) world.energy = worldS;
            // 4. materials
            for (int i = 0; i < 72; i++)
            {
                float u = i / 72f;
                if (paint != null) { paint.baseColor = Color.HSVToRGB(u, 0.85f, 0.7f); paint.roughness = Mathf.Clamp(paintR + 0.4f * Mathf.Sin(u * Mathf.PI * 2f), 0.03f, 1f); }
                if (brushed != null) brushed.roughness = Mathf.Clamp(brushedR - 0.3f * Mathf.Sin(u * Mathf.PI), 0.03f, 1f);
                Shoot("materials");
            }
            File.WriteAllText(Path.Combine(outDir, "frames.txt"), log.ToString());
            Debug.Log("NR_CAPTURE_OK " + n + " frames -> " + outDir);
            nr.Unload(); UnityEngine.Object.DestroyImmediate(go);
            EditorApplication.Exit(0);
        }
    }
}
