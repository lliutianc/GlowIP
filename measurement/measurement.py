class Measurement:
    def __call__(self, images):
        return self.forward(images)

    def forward(self, images):
        pass

