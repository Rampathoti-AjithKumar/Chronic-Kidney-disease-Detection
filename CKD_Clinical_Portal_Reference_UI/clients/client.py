from models.ckd_model import (
    create_model,
    train_model,
    evaluate_model,
    get_model_parameters,
    set_model_parameters
)


class CKDClient:

    def __init__(
        self,
        client_id,
        X_train,
        y_train,
        X_validation,
        y_validation
    ):

        self.client_id = client_id

        # Local training data
        self.X_train = X_train
        self.y_train = y_train

        # Local validation data
        self.X_validation = X_validation
        self.y_validation = y_validation

        # Create local CKD model
        self.model = create_model()


    def train(self):

        self.model = train_model(
            self.model,
            self.X_train,
            self.y_train
        )

        print(
            f"{self.client_id}: "
            f"Local model training completed."
        )


    def evaluate(self):

        metrics = evaluate_model(
            self.model,
            self.X_validation,
            self.y_validation
        )

        return metrics


    def get_parameters(self):

        return get_model_parameters(
            self.model
        )


    def set_parameters(self, parameters):

        self.model = set_model_parameters(
            self.model,
            parameters
        )


    def get_num_samples(self):

        return len(self.X_train)