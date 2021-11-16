def plot_persistence_landscapes(encoder, decoder, image_set, num_landscapes=3, num_points=100):
    points = get_encoded_points_and_labels(encoder, decoder, image_set, num_points)[0]
    simplex_tree = make_simplicial_complex(points)
    LS = gd.representations.Landscape(num_landscapes=num_landscapes, resolution=num_points)
    L = LS.fit_transform([simplex_tree.persistence_intervals_in_dimension(1)])
    for i in range(num_landscapes):
        plt.plot(L[0][i * num_points:(i+1) * num_points])
    plt.show()
    # plot_encoded_points(np.array(points), labels, image_set)


def make_simplicial_complex(points):
    skeleton = gd.RipsComplex(points=points, max_edge_length=1.7)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=2)
    barcodes = simplex_tree.persistence()
    #     for barcode in barcodes:
    #         print(barcode)
    return simplex_tree


def get_encoded_points_and_labels(encoder, decoder, image_set, num_points):
    points = []
    labels = []
    for i in range(num_points):
        img = image_set[i][0].unsqueeze(0).to(device)
        label = image_set[i][1]
        labels.append(label)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            encoded_img = encoder(img)
            rec_img = decoder(encoded_img)
        encoded_img = encoded_img.flatten().cpu().numpy()
        points.append(encoded_img)
    return points, labels