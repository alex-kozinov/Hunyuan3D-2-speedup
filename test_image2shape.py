from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

def main():
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
    result = pipe(image="assets/demo.png")
    mesh = result[0]

    mesh.export("outputs/demo_mesh.glb")
    print("Saved mesh to outputs/demo_mesh.glb")

if __name__ == "__main__":
    main()
