// Primary visibility: every triangle of this frame's table drawn with its index, depth-tested.
Shader "Hidden/NeuronRender/Id" {
    SubShader {
        Tags { "RenderType" = "Opaque" }
        Pass {
            Cull Off ZWrite On ZTest LEqual
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            StructuredBuffer<float> _Tri;
            float4x4 _NeuronVP;
            struct v2f { float4 pos : SV_POSITION; nointerpolation int tri : TEXCOORD0; };
            v2f vert(uint vid : SV_VertexID) {
                v2f o;
                uint t = vid / 3u, c = vid - 3u * t;
                uint b = t * 24u + c * 3u;
                float3 p = float3(_Tri[b], _Tri[b + 1u], _Tri[b + 2u]);
                o.pos = mul(_NeuronVP, float4(p, 1.0));
                o.tri = int(t);
                return o;
            }
            float frag(v2f i) : SV_Target { return float(i.tri); }
            ENDHLSL
        }
    }
}
