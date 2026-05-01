class Config:
    # trian config
    epochs = 100
    save_model = True

    # model config
    batch = 64
    num_head = 8
    num_embed = 64
    head = 16 * num_head
    kernel = 2
    num_layer = 12
    dropout = 0.1
    learning_rate = 5e-5

    def print_config(self):
        for name, value in self.__class__.__dict__.items():
            if name in {
                "epochs",
                "batch",
                "num_head",
                "num_embed",
                "head",
                "kernel",
                "num_layer",
                "dropout",
                "learning_rate"
            }:
                print(f"{name:<20}: \033[1;92m{value}\033[0m")
        print("\n")
