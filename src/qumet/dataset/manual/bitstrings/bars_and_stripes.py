def bars_and_stripes(rows, cols):
    data = []

    for h in itertools.product([0, 1], repeat=cols):
        pic = np.repeat([h], rows, 0)
        data.append(pic.ravel().tolist())

    for h in itertools.product([0, 1], repeat=rows):
        pic = np.repeat([h], cols, 1)
        data.append(pic.ravel().tolist())

    data = np.unique(np.asarray(data), axis=0)

    return data


# Example usage (commented out to avoid execution on import)
# if __name__ == "__main__":
#     n, m = 2, 3
#     bas = bars_and_stripes(n, m)
#     print(bas)
#     n_points, n_qubits = bas.shape
#     print(n_points, n_qubits)
#     fig, ax_b = plt.subplots(1, bas.shape[0], figsize=(14, 2))
#     for i in range(bas.shape[0]):
#         ax_b[i].matshow(bas[i].reshape(n, m), vmin=-1, vmax=1)
#         ax_b[i].set_xticks([])
#         ax_b[i].set_yticks([])
#         ax_b[i].set_xticks([0.5], minor=True)
#         ax_b[i].set_yticks([0.5], minor=True)
#         ax_b[i].set_title(bas[i])
#         ax_b[i].grid(which="minor", color="black", linestyle="-", linewidth=0.75)
#     plt.show()
