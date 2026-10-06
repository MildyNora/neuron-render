// A small on-screen panel for a NeuronRenderer: frame scrubber, one row per light, one per editable material.
using UnityEngine;

namespace NeuronRender
{
    [RequireComponent(typeof(NeuronRenderer))]
    public class NeuronDemoUI : MonoBehaviour
    {
        NeuronRenderer nr; Vector2 scroll; bool show = true; float fps; int frames; float tAcc;
        public bool playAnimation = true; public float playbackFps = 24f; float playClock;

        void Update()
        {
            nr = nr ?? GetComponent<NeuronRenderer>();
            if (Input.GetKeyDown(KeyCode.Tab)) show = !show;
            if (Input.GetKeyDown(KeyCode.Space)) playAnimation = !playAnimation;
            tAcc += Time.unscaledDeltaTime; frames++;
            if (tAcc > 0.5f) { fps = frames / tAcc; frames = 0; tAcc = 0; }
            if (playAnimation && nr.Scene != null && nr.Scene.FrameCount > 1)
            {
                playClock += Time.unscaledDeltaTime * playbackFps;
                nr.frame = Mathf.FloorToInt(playClock) % nr.Scene.FrameCount;
            }
        }

        void OnGUI()
        {
            if (!show || nr == null) return;
            GUI.skin.label.fontSize = 13;
            GUILayout.BeginArea(new Rect(12, 12, 360, Screen.height - 24), GUI.skin.box);
            GUILayout.Label(string.Format("Neuron Render  {0:F1} fps  ({1:F1} ms)   [Tab] hide  [Space] play/pause", fps, nr.LastFrameMs));
            nr.preset = (NeuronRenderer.Preset)GUILayout.SelectionGrid((int)nr.preset, new[] { "draft", "fast", "balanced", "high", "ultra" }, 5);
            if (nr.Scene != null && nr.Scene.FrameCount > 1)
            {
                GUILayout.Label("frame " + (nr.frame + 1) + " / " + nr.Scene.FrameCount);
                int f = Mathf.RoundToInt(GUILayout.HorizontalSlider(nr.frame, 0, nr.Scene.FrameCount - 1));
                if (f != nr.frame) { nr.frame = f; playClock = f; playAnimation = false; }
            }
            scroll = GUILayout.BeginScrollView(scroll);
            foreach (var l in nr.lights)
            {
                GUILayout.Label(l.isWorld ? l.name + "  strength " + l.energy.ToString("F2") : l.name + "  energy " + l.energy.ToString("F1"));
                float max = l.isWorld ? 3f : Mathf.Max(l.energy * 3f, 1f);
                l.energy = GUILayout.HorizontalSlider(l.energy, 0f, max);
                if (!l.isWorld) l.color = ColorRow(l.color);
            }
            foreach (var m in nr.materials)
            {
                GUILayout.Label(m.name + "  roughness " + m.roughness.ToString("F2"));
                m.roughness = GUILayout.HorizontalSlider(m.roughness, 0.03f, 1f);
                if (m.colorEditable) m.baseColor = ColorRow(m.baseColor);
            }
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        static Color ColorRow(Color c)
        {
            GUILayout.BeginHorizontal();
            c.r = GUILayout.HorizontalSlider(c.r, 0f, 1f); c.g = GUILayout.HorizontalSlider(c.g, 0f, 1f); c.b = GUILayout.HorizontalSlider(c.b, 0f, 1f);
            var old = GUI.color; GUI.color = new Color(Mathf.Pow(c.r, 0.45f), Mathf.Pow(c.g, 0.45f), Mathf.Pow(c.b, 0.45f));
            GUILayout.Box("", GUILayout.Width(22), GUILayout.Height(18)); GUI.color = old;
            GUILayout.EndHorizontal();
            return c;
        }
    }
}
