import horovod.torch as hvd

if __name__ == "__main__":
    # test if horovod is correct installed
    hvd.init()
    print("Hello from rank{}".format(hvd.rank()))