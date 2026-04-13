using UnityEngine;

public class GPUFluidSimulation : MonoBehaviour
{
    public ComputeShader computeShader;

    RenderTexture densityCurrent;
    RenderTexture densityPrevious;

    RenderTexture velocityCurrent;
    RenderTexture velocityPrevious;

    Vector2 lastMouseUV;

    void Start()
    {
        densityCurrent = CreateTexture();
        densityPrevious = CreateTexture();

        velocityCurrent = CreateTexture();
        velocityPrevious = CreateTexture();

        lastMouseUV = new Vector2(0.5f, 0.5f);
    }

    RenderTexture CreateTexture()
    {
        RenderTexture rt = new RenderTexture(512, 512, 0);
        rt.enableRandomWrite = true;
        rt.Create();
        return rt;
    }

    void Update()
    {
        int kernel = computeShader.FindKernel("CSMain");

        Vector3 mouse = Input.mousePosition;
        Vector2 mouseUV = new Vector2(mouse.x / Screen.width, mouse.y / Screen.height);

        Vector2 mouseDelta = mouseUV - lastMouseUV;
        lastMouseUV = mouseUV;

        computeShader.SetTexture(kernel, "DensityResult", densityCurrent);
        computeShader.SetTexture(kernel, "DensityPrevious", densityPrevious);

        computeShader.SetTexture(kernel, "VelocityResult", velocityCurrent);
        computeShader.SetTexture(kernel, "VelocityPrevious", velocityPrevious);

        computeShader.SetVector("MousePos", mouseUV);
        computeShader.SetVector("MouseDelta", mouseDelta);
        computeShader.SetFloat("Radius", 20.0f);

        computeShader.Dispatch(kernel, 512 / 8, 512 / 8, 1);

        Swap(ref densityCurrent, ref densityPrevious);
        Swap(ref velocityCurrent, ref velocityPrevious);
    }

    void Swap(ref RenderTexture a, ref RenderTexture b)
    {
        RenderTexture temp = a;
        a = b;
        b = temp;
    }

    public RenderTexture GetTexture()
    {
        return densityCurrent;
    }
}