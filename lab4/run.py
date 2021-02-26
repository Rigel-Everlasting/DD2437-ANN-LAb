from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet

if __name__ == "__main__":
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''

    print("\nStarting a Restricted Boltzmann Machine..")

    # rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
    #                                  ndim_hidden=500,
    #                                  is_bottom=True,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=10
    #                                  )

    # rbm.cd1(visible_trainset=train_imgs, n_iterations=20, plot=True, plot_title=f"hidden nodes={rbm.ndim_hidden}",
    #         visualize_w=True)

    rbm_200 = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=10
    )

    rbm_500 = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=10
    )
    # RBM 200
    rbm_200_errors_per_epoch = rbm_200.cd1(visible_trainset=train_imgs, n_iterations=20, visualize_w=False)
    pics = test_imgs[0:10][:]
    rbm_200.visualize_reconstruction(pics, image_size)
    input('Press to continue')
    recon_errors = rbm_200.compute_reconstruction_errors(test_imgs, image_size)
    print('RBM 200 TEST RECON ERROR: ', np.mean(recon_errors))

    # RBM 500
    rbm_500_errors_per_epoch = rbm_500.cd1(visible_trainset=train_imgs, n_iterations=20, visualize_w=False)
    pics = test_imgs[0:10][:]
    rbm_200.visualize_reconstruction(pics, image_size)
    # input('Press to continue')
    recon_errors = rbm_200.compute_reconstruction_errors(test_imgs, image_size)
    print('RBM 500 TEST RECON ERROR: ', np.mean(recon_errors))

    input('done?')
    input('Are you really done?')

    ''' deep- belief net '''
    #
    # print("\nStarting a Deep Belief Net..")
    #
    # dbn = DeepBeliefNet(sizes={"vis": image_size[0] * image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
    #                     image_size=image_size,
    #                     n_labels=10,
    #                     batch_size=10
    #                     )
    #
    # ''' greedy layer-wise training '''
    #
    # dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=15)
    #
    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)

    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1, 10))
    #     digit_1hot[0, digit] = 1
    #     r = dbn.generate(digit_1hot, name="rbms")

    #
    # ''' fine-tune wake-sleep training '''
    #
    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10)
    #
    # dbn.recognize(train_imgs, train_lbls)
    #
    # dbn.recognize(test_imgs, test_lbls)
    #
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1, 10))
    #     digit_1hot[0, digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
