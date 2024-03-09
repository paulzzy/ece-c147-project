import torch


class EEGClassifier(torch.nn.Module):
    """
    Uses CNN layers to classify EEG data. Input data should have shape
    (batch_size, 22, 1000), since there are 22 electrodes and 1000 time bins.
    Output data should have shape (batch_size, 1) where the class labels are in
    [769, 770, 771, 772].
    """

    ELECTRODES = 22
    TIME_BINS = 1000
    CLASSES = [769, 770, 771, 772]

    def __init__(self):
        super(EEGClassifier, self).__init__()

        # TODO

    def forward(self, x):
        # TODO
        return x

    def evaluate(self, x, y):
        with torch.no_grad():
            y_pred = self.forward(x)
            y_pred = torch.argmax(y_pred, dim=1)
            accuracy = torch.sum(y_pred == y).item() / len(y)
            return accuracy

    def predict(self, x):
        with torch.no_grad():
            y_pred = self.forward(x)
            y_pred = torch.argmax(y_pred, dim=1)
            return y_pred

    def loss(self, y_pred, y):
        loss = torch.nn.CrossEntropyLoss()
        return loss(y_pred, y)

    def fit(
        self, X_train, y_train, X_valid, y_valid, epochs=10, batch_size=32, lr=0.001
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                x_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]
                y_pred = self.forward(x_batch)
                loss = self.loss(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_accuracy = self.evaluate(X_train, y_train)
            valid_accuracy = self.evaluate(X_valid, y_valid)
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Accuracy: {train_accuracy:.2f}, Valid Accuracy: {valid_accuracy:.2f}"
            )
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self
