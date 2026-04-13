using UnityEngine;

public class DisplayTexture : MonoBehaviour
{
    public GPUFluidSimulation sim;

    void Update()
    {
        GetComponent<Renderer>().material.mainTexture = sim.GetTexture();
    }
}