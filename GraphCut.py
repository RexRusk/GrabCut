import numpy as np
import cv2
import maxflow
def load_image(image_path):
    return cv2.imread(image_path)
def graph_cut(image, mask):
    # Convert the input image to grayscale if it's a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a graph
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(image.shape)

    # Define the unary term (data cost)
    data_cost = 1 - np.abs(image.astype(np.float32) - mask.astype(np.float32)) / 255.0

    # Add the unary term to the graph
    g.add_grid_tedges(nodeids, data_cost, 1 - data_cost)

    # Add the pairwise term (smoothness cost) - in this case, we use the 2D grid graph
    g.add_grid_edges(nodeids, 1)

    # Find the minimum cut on the graph
    g.maxflow()

    # Get the resulting segmentation
    seg = g.get_grid_segments(nodeids)

    # Return the binary mask representing the segmented regions
    return seg.astype(np.uint8)
if __name__ == "__main__":
    image_path = "input_image.jpg"
    mask_path = "input_mask.jpg"

    # Load the image and mask
    image = load_image(image_path)
    mask = load_image(mask_path)

    # Perform graph cut
    segmented_mask = graph_cut(image, mask)

    # Show the segmented result
    cv2.imshow("Segmented Image", segmented_mask * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
