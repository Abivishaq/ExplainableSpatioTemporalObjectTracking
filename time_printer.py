from helpers.encoders import *

try:
    while True:
        t = input("enter time:")
        print(human_readable_from_external(torch.Tensor([float(t)])))
except KeyboardInterrupt:
    print("\nExiting...")
    