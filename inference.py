from models.nmlnet import create_nml_model, nmlcls

if __name__ == "__main__":

    video_path = "test.mp4"

    cls_model = create_nml_model()
    g_nmlnet_cls = nmlcls(cls_model, video_path)
    print(g_nmlnet_cls)