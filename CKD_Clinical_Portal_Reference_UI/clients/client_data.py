import numpy as np


def create_client_datasets(
    X_train,
    y_train,
    num_clients=3,
    validation_size=0.20
):

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Reproducible client assignment for consistent experiments.
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_train))

    X_train = X_train[indices]
    y_train = y_train[indices]

    client_indices = np.array_split(
        np.arange(len(X_train)),
        num_clients
    )

    clients = {}

    for i, indices in enumerate(client_indices):

        X_client = X_train[indices]
        y_client = y_train[indices]

        split_index = int(
            len(X_client) * (1 - validation_size)
        )

        X_local_train = X_client[:split_index]
        y_local_train = y_client[:split_index]

        X_local_validation = X_client[split_index:]
        y_local_validation = y_client[split_index:]

        clients[f"client_{i + 1}"] = {

            "X_train": X_local_train,
            "y_train": y_local_train,

            "X_validation": X_local_validation,
            "y_validation": y_local_validation
        }

    return clients


def show_client_distribution(clients):

    print("\nClient Data Distribution:")

    for client_name, data in clients.items():

        X_train = data["X_train"]
        y_train = data["y_train"]

        X_validation = data["X_validation"]
        y_validation = data["y_validation"]

        print(
            f"\n{client_name}:"
        )

        print(
            f"  Training samples: "
            f"{len(X_train)}"
        )

        print(
            f"  Validation samples: "
            f"{len(X_validation)}"
        )

        print(
            f"  CKD training samples: "
            f"{np.sum(y_train == 0)}"
        )

        print(
            f"  Not CKD training samples: "
            f"{np.sum(y_train == 1)}"
        )


if __name__ == "__main__":

    from preprocessing.preprocess import prepare_data

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        label_encoders,
        target_encoder
    ) = prepare_data()

    clients = create_client_datasets(
        X_train,
        y_train,
        num_clients=3
    )

    show_client_distribution(clients)