import matplotlib
import datetime
import os

def generate_images(model, test_input, target, output_path):
    img = model(test_input)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], target[0], img[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    
    plt.savefig(os.path.join(output_path, "{}_result.png".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))))
    plt.show()
    