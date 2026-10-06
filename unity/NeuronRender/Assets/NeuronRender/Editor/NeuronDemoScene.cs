// Builds Assets/Scenes/NeuronDemo.unity: a camera at the pack's own first-frame pose, carrying the renderer and the panel.
//   Unity -batchmode -projectPath unity/NeuronRender -executeMethod NeuronRender.Editor.NeuronDemoScene.Build -quit -nrpack <pack dir>
using System;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace NeuronRender.Editor
{
    public static class NeuronDemoScene
    {
        static string Arg(string name, string fallback = null)
        {
            var a = Environment.GetCommandLineArgs();
            for (int i = 0; i < a.Length - 1; i++) if (a[i] == name) return a[i + 1];
            return fallback;
        }

        [MenuItem("Neuron Render/Build demo scene...")]
        public static void BuildFromMenu()
        {
            string pack = EditorUtility.OpenFolderPanel("Pack directory (contains pack.json)", "", "");
            if (!string.IsNullOrEmpty(pack)) Build(pack);
        }

        public static void Build() { Build(Arg("-nrpack")); if (Application.isBatchMode) EditorApplication.Exit(0); }

        public static void Build(string packPath)
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var go = new GameObject("Neuron Camera");
            var cam = go.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor; cam.backgroundColor = Color.black; cam.nearClipPlane = 0.05f; cam.farClipPlane = 1000f;
            var nr = go.AddComponent<NeuronRenderer>();
            nr.packPath = packPath;
            go.AddComponent<NeuronDemoUI>();
            if (!string.IsNullOrEmpty(packPath) && System.IO.File.Exists(System.IO.Path.Combine(packPath, "pack.json")))
            {
                var meta = JObject.Parse(System.IO.File.ReadAllText(System.IO.Path.Combine(packPath, "pack.json")));
                var m = meta["cameras"][0]["matrix"];
                // Blender camera-to-world -> Unity transform (swap y and z; Unity's camera looks along +z, Blender's along -z)
                var right = new Vector3((float)m[0][0], (float)m[2][0], (float)m[1][0]);
                var up = new Vector3((float)m[0][1], (float)m[2][1], (float)m[1][1]);
                var back = new Vector3((float)m[0][2], (float)m[2][2], (float)m[1][2]);
                go.transform.position = new Vector3((float)m[0][3], (float)m[2][3], (float)m[1][3]);
                go.transform.rotation = Quaternion.LookRotation(-back, up);
                float lens = (float)meta["cameras"][0]["lens"], sw = (float)meta["sensor_width"], sh = (float)meta["sensor_height"];
                int w = (int)meta["width"], h = (int)meta["height"];
                string fit = (string)meta["sensor_fit"];
                bool horizontal = fit == "HORIZONTAL" || (fit != "VERTICAL" && w >= h);
                float sensor = fit == "VERTICAL" ? sh : sw, viewfac = horizontal ? w : h, f = lens / sensor * viewfac;
                cam.fieldOfView = 2f * Mathf.Atan(0.5f * h / f) * Mathf.Rad2Deg;
            }
            System.IO.Directory.CreateDirectory("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, "Assets/Scenes/NeuronDemo.unity");
            Debug.Log("NR_SCENE_OK Assets/Scenes/NeuronDemo.unity -> " + packPath);
        }
    }
}
