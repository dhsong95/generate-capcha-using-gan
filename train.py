from network import GANNetwork


if __name__ == "__main__":
    network = GANNetwork()
    network.train(epochs=3000000, batch_size=32, save_interval=50)
