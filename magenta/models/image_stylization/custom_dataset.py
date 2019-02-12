import os

class CustomDataset():

    def __init__(self, path):

        self.path = path
        self.labels = []
        self.idx = 0
        self.epochs = 0

        for filename in os.listdir(path + 'img/train_legible/'):
            self.labels.append(path + 'img/train_legible/' + filename)

        print("Total elements: " + str(len(self.labels)))

    def len(self):
        return len(self.labels)

    def get_item(self):

        img_filename = self.labels[self.idx]
        mask_filename = self.labels[self.idx].replace('/img/','/gt/').replace('/train_legible/','/masks_legible_trainval/').replace('.jpg','.png')

        self.idx += 1
        if self.idx >= len(self.labels):
            self.idx = 0
            self.epochs += 1

        if self.idx % 16 == 0:
            print("Epoch " + str(self.epochs) + "  Iter " + str(self.idx))

        return img_filename, mask_filename