from sklearn.cluster import KMeans

"""
TeamAssigner Class
-----------------
This class assigns players to teams based on shirt color using K-means clustering.
It identifies two distinct team colors from player images and classifies each
player accordingly.

The algorithm works by:
1. Extracting the top half of each player bounding box (where the shirt is typically located)
2. Applying K-means clustering with 2 clusters to separate player shirt from background
3. Identifying which cluster represents the player by comparing with corner pixels
4. Using a second K-means clustering to group player colors into two teams
"""
class TeamAssigner:
    def __init__(self):
        """
        Initialize the TeamAssigner with empty dictionaries to store team colors
        and player team assignments.
        
        Attributes:
            team_colors (dict): Maps team IDs to their representative RGB colors
            player_team (dict): Maps player IDs to their assigned team IDs
        """
        self.team_colors = {}
        self.player_team = {}
        # Note: Unnecessary 'pass' statement

    def get_clustering_model(self, image):
        """
        Create and fit a K-means clustering model on the image data.
        
        Args:
            image (numpy.ndarray): RGB image array with shape (height, width, 3)
            
        Returns:
            sklearn.cluster.KMeans: Fitted K-means model with 2 clusters
            
        Notes:
            - Reshapes the 3D image array to 2D array of RGB values
            - Uses k-means++ initialization for better convergence
            - Uses n_init=1 for performance (may impact stability)
        """
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)

        return kmeans 

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant color (RGB values) of a player's shirt from a frame.
        
        Args:
            frame (numpy.ndarray): Full video frame containing the player
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            numpy.ndarray: RGB color values representing the player's shirt
            
        Algorithm:
            1. Extract the player image using the bounding box
            2. Focus on the top half (where shirt is typically located)
            3. Apply K-means clustering to separate player from background
            4. Identify background cluster by checking corner pixels
            5. Return the center of the non-background cluster as player color
        """
        # Extract player image using bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  
        # Focus on top half of player (where the shirt is)
        top_half = image[0: int(image.shape[0]/2), :]

        # Apply K-means clustering to separate player from background
        kmeans = self.get_clustering_model(top_half)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to match the image dimensions
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])

        # Identify background cluster by sampling the four corners
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], 
                          clustered_image[-1,0], clustered_image[-1,-1]]

        # The most common cluster in corners is likely the background
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        # Player cluster is the other cluster (assuming 2 clusters total)
        player_cluster = 1 - non_player_cluster

        # Get the RGB center of the player cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_colors(self, frame, player_detections):
        """
        Determine the two team colors based on all detected players in a frame.
        
        Args:
            frame (numpy.ndarray): Video frame containing all players
            player_detections (dict): Dictionary of player detections with format
                                     {player_id: {"bbox": [x1, y1, x2, y2], ...}}
        
        Side Effects:
            - Updates self.kmeans with the fitted clustering model
            - Updates self.team_colors with the two team color centroids
            
        Algorithm:
            1. Extract dominant shirt color for each player
            2. Use K-means with 2 clusters to separate all players into two teams
            3. Store the cluster centers as team colors
        """
        player_colors = []

        # Extract the shirt color for each detected player
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # Cluster player colors into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        # Store the K-means model for later prediction
        self.kmeans = kmeans

        # Assign team IDs (1 and 2) to the cluster centers
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determine the team assignment for a specific player.
        
        Args:
            frame (numpy.ndarray): Video frame containing the player
            player_bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            player_id (any): Unique identifier for the player
            
        Returns:
            int: Team ID (1 or 2)
            
        Side Effects:
            - Updates self.player_team dictionary with the player's team assignment
            
        Notes:
            - Returns cached result if player_id already has a team assigned
            - Otherwise extracts player color and uses the pre-trained model to predict team
        """
        # Return cached team assignment if available
        if player_id in self.player_team:
            return self.player_team[player_id]

        # Extract player shirt color
        player_color = self.get_player_color(frame, player_bbox)

        # Predict team using the K-means model
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        # Adjust to use team IDs 1 and 2 (instead of 0 and 1)
        team_id += 1

        # Cache the team assignment for future use
        self.player_team[player_id] = team_id

        return team_id