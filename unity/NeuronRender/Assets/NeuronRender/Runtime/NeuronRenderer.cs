// Drop on a Camera: the camera's image becomes the fitted scene, rendered by the network each frame.
using System.Collections.Generic;
using UnityEngine;

namespace NeuronRender
{
    [RequireComponent(typeof(Camera))]
    [ExecuteAlways]
    public class NeuronRenderer : MonoBehaviour
    {
        [Tooltip("Directory written by `neuron-render export` (contains pack.json)")]
        public string packPath;
        public enum Preset { Draft, Fast, Balanced, High, Ultra }
        public Preset preset = Preset.Balanced;
        [Tooltip("Animation frame, for fits that cover time")]
        public int frame;
        [System.Serializable] public class LightSetting { public string name; public float energy = 1; public Color color = Color.white; public bool isWorld; }
        [System.Serializable] public class MaterialSetting { public string name; public Color baseColor = Color.white; [Range(0, 1)] public float roughness = 0.5f; public bool colorEditable; }
        public List<LightSetting> lights = new List<LightSetting>();
        public List<MaterialSetting> materials = new List<MaterialSetting>();
        public bool flipId = true, flipOutput = true;

        NeuronPack pack; NeuronScene scene; string loadedPath;
        public NeuronScene Scene => scene;
        public float LastFrameMs { get; private set; }

        public bool Load()
        {
            if (scene != null && loadedPath == packPath) return true;
            Unload();
            if (string.IsNullOrEmpty(packPath) || !System.IO.File.Exists(System.IO.Path.Combine(packPath, "pack.json"))) return false;
            pack = new NeuronPack(packPath);
            scene = new NeuronScene(pack);
            loadedPath = packPath;
            if (lights.Count == 0 && pack.Parametric)
                foreach (var g in pack.Groups)
                {
                    string kind = (string)g["kind"];
                    var ls = new LightSetting { name = (string)g["name"], isWorld = kind != "light" };
                    if (kind == "light") { ls.energy = (float)g["energy"]; var c = g["color"].ToObject<float[]>(); ls.color = new Color(c[0], c[1], c[2]); }
                    else ls.energy = g["strength"] != null ? (float)g["strength"] : 1f;
                    lights.Add(ls);
                }
            if (materials.Count == 0)
                foreach (var m in pack.Materials)
                {
                    var editable = m["editable"].ToObject<string[]>();
                    if (editable.Length == 0) continue;
                    var c = m["base_color"].ToObject<float[]>();
                    materials.Add(new MaterialSetting { name = (string)m["name"], baseColor = new Color(c[0], c[1], c[2]), roughness = (float)m["roughness"],
                                                        colorEditable = System.Array.IndexOf(editable, "base_color") >= 0 });
                }
            return true;
        }

        public void Unload() { scene?.Dispose(); pack?.Dispose(); scene = null; pack = null; loadedPath = null; }
        void OnDisable() => Unload();

        NeuronQuality QualityOf(Preset p) => p switch
        {
            Preset.Draft => NeuronQuality.Draft, Preset.Fast => NeuronQuality.Fast, Preset.High => NeuronQuality.High, Preset.Ultra => NeuronQuality.Ultra,
            _ => NeuronQuality.Balanced,
        };

        void Apply()
        {
            scene.Quality = QualityOf(preset);
            scene.Frame = frame;
            scene.FlipId = flipId; scene.FlipOut = flipOutput;
            scene.LightColor.Clear(); scene.LightEnergy.Clear();
            foreach (var l in lights) { scene.LightEnergy[l.name] = l.energy; if (!l.isWorld) scene.LightColor[l.name] = new Vector3(l.color.r, l.color.g, l.color.b); }
            scene.MaterialColor.Clear(); scene.MaterialRoughness.Clear();
            foreach (var m in materials)
            {
                scene.MaterialRoughness[m.name] = m.roughness;
                if (m.colorEditable) scene.MaterialColor[m.name] = new Vector3(m.baseColor.r, m.baseColor.g, m.baseColor.b);
            }
            scene.MarkMaterialsDirty();
        }

        void OnRenderImage(RenderTexture src, RenderTexture dst)
        {
            if (!Load()) { Graphics.Blit(src, dst); return; }
            Apply();
            var cam = GetComponent<Camera>();
            var t0 = System.Diagnostics.Stopwatch.StartNew();
            scene.Render(NeuronCamera.FromUnity(cam, dst != null ? dst.width : cam.pixelWidth, dst != null ? dst.height : cam.pixelHeight), dst);
            LastFrameMs = (float)t0.Elapsed.TotalMilliseconds;
        }
    }
}
