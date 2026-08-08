# Project-
# ================================================================
# WATER BODY DETECTION AND ANALYSIS FROM SATELLITE JPG/PNG
# ================================================================
#
# INPUT:
#   Satellite JPG / JPEG / PNG
#
# OUTPUT:
#   1. Original satellite image
#   2. Water probability / segmentation mask
#   3. Clean water mask
#   4. Detected water body
#   5. Blue transparent water overlay
#   6. Red water boundary
#   7. Yellow bounding box
#   8. Center point
#   9. Water pixel area
#  10. Water coverage percentage
#  11. Perimeter
#  12. Width / Height
#  13. Aspect ratio
#  14. Compactness
#  15. CSV analysis report
#  16. Optional before/after change analysis
#
# Google Colab - SINGLE CELL
# ================================================================


# ------------------------------------------------
# 1. INSTALL LIBRARIES
# ------------------------------------------------

import sys
!{sys.executable} -m pip install -q opencv-python matplotlib numpy pandas


# ------------------------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

from google.colab import files


# ================================================================
# FUNCTION 1:
# WATER DETECTION AND ANALYSIS
# ================================================================

def detect_water(image_path):

    print("\n")
    print("=" * 70)
    print("PROCESSING:", image_path)
    print("=" * 70)

    # ------------------------------------------------------------
    # READ IMAGE
    # ------------------------------------------------------------

    img = cv2.imread(image_path)

    if img is None:
        raise Exception(
            "Image could not be loaded. Please upload JPG/PNG."
        )

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    original = rgb.copy()

    height, width = rgb.shape[:2]

    total_pixels = height * width

    print("Image size:", width, "x", height)


    # ------------------------------------------------------------
    # RESIZE LARGE IMAGES
    # ------------------------------------------------------------

    max_dimension = 1600

    scale = 1.0

    if max(height, width) > max_dimension:

        scale = max_dimension / max(height, width)

        new_width = int(width * scale)
        new_height = int(height * scale)

        rgb = cv2.resize(
            rgb,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        height, width = rgb.shape[:2]

        print(
            "Processing size:",
            width,
            "x",
            height
        )


    # ============================================================
    # 3. COLOR SPACE CONVERSION
    # ============================================================

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV
    )

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )


    # ------------------------------------------------------------
    # RGB CHANNELS
    # ------------------------------------------------------------

    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)


    # ------------------------------------------------------------
    # HSV CHANNELS
    # ------------------------------------------------------------

    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)


    # ------------------------------------------------------------
    # LAB CHANNELS
    # ------------------------------------------------------------

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    LAB_B = lab[:, :, 2].astype(np.float32)


    # ============================================================
    # 4. CREATE WATER PROBABILITY MAP
    # ============================================================

    # Water often has:
    # - relatively low brightness
    # - blue/cyan characteristics
    # - lower red contribution
    #
    # We combine several indicators instead of using
    # only G-R as in the previous code.

    blue_score = B / (R + G + B + 1.0)

    red_suppression = 1.0 - (
        R / (R + G + B + 1.0)
    )

    darkness_score = 1.0 - (
        V / 255.0
    )

    saturation_score = (
        S / 255.0
    )


    # Normalize each score between 0 and 1

    blue_score = np.clip(
        blue_score,
        0,
        1
    )

    red_suppression = np.clip(
        red_suppression,
        0,
        1
    )

    darkness_score = np.clip(
        darkness_score,
        0,
        1
    )

    saturation_score = np.clip(
        saturation_score,
        0,
        1
    )


    # Combined probability score

    probability = (
        0.40 * blue_score +
        0.25 * red_suppression +
        0.20 * darkness_score +
        0.15 * saturation_score
    )

    probability = np.clip(
        probability,
        0,
        1
    )


    # ============================================================
    # 5. CREATE INITIAL WATER MASK
    # ============================================================

    # Adaptive threshold based on image

    threshold_value = np.percentile(
        probability,
        75
    )

    # Keep threshold within reasonable range

    threshold_value = np.clip(
        threshold_value,
        0.45,
        0.70
    )

    initial_mask = (
        probability >= threshold_value
    ).astype(np.uint8) * 255


    # ============================================================
    # 6. REMOVE EXTREME DARK SHADOWS
    # ============================================================

    # Very dark pixels can be buildings, roads or shadows.
    # Water should generally have some color information.

    not_extreme_black = (
        (R + G + B) > 45
    )

    initial_mask[
        ~not_extreme_black
    ] = 0


    # ============================================================
    # 7. MORPHOLOGICAL CLEANING
    # ============================================================

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    clean_mask = cv2.morphologyEx(
        initial_mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    clean_mask = cv2.morphologyEx(
        clean_mask,
        cv2.MORPH_CLOSE,
        kernel_large,
        iterations=2
    )


    # ============================================================
    # 8. CONNECTED COMPONENT ANALYSIS
    # ============================================================

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            clean_mask,
            connectivity=8
        )


    # ------------------------------------------------------------
    # Remove very small regions
    # ------------------------------------------------------------

    minimum_area = max(
        100,
        int(height * width * 0.0002)
    )

    filtered_mask = np.zeros_like(
        clean_mask
    )

    components = []

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_area:

            filtered_mask[
                labels == i
            ] = 255

            components.append(
                (i, area)
            )


    # ============================================================
    # 9. SELECT MOST LIKELY WATER BODY
    # ============================================================

    if len(components) == 0:

        raise Exception(
            "No water-like region detected. "
            "Try a clearer satellite image."
        )


    # Sort by area

    components.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # ------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------

    selected_label = None

    image_center_x = width / 2
    image_center_y = height / 2


    for label_id, area in components[:10]:

        cx = centroids[
            label_id
        ][0]

        cy = centroids[
            label_id
        ][1]

        distance = math.sqrt(
            (
                cx - image_center_x
            ) ** 2 +
            (
                cy - image_center_y
            ) ** 2
        )

        normalized_distance = (
            distance /
            math.sqrt(
                width ** 2 +
                height ** 2
            )
        )

        # Prefer large regions that are not extremely
        # far from the image center.

        if normalized_distance < 0.45:

            selected_label = label_id
            break


    # If no central region found, use largest

    if selected_label is None:

        selected_label = components[0][0]


    # ============================================================
    # 10. FINAL WATER MASK
    # ============================================================

    final_mask = np.zeros_like(
        filtered_mask
    )

    final_mask[
        labels == selected_label
    ] = 255


    # ============================================================
    # 11. FIND CONTOURS
    # ============================================================

    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) == 0:

        raise Exception(
            "Could not find water boundary."
        )


    water_contour = max(
        contours,
        key=cv2.contourArea
    )


    # ============================================================
    # 12. WATER AREA
    # ============================================================

    water_pixel_area = cv2.contourArea(
        water_contour
    )

    water_mask_pixels = np.sum(
        final_mask == 255
    )


    # ============================================================
    # 13. WATER COVERAGE
    # ============================================================

    coverage_percentage = (
        water_mask_pixels /
        (height * width)
    ) * 100


    # ============================================================
    # 14. BOUNDING BOX
    # ============================================================

    x, y, w, h = cv2.boundingRect(
        water_contour
    )


    # ============================================================
    # 15. PERIMETER
    # ============================================================

    perimeter = cv2.arcLength(
        water_contour,
        True
    )


    # ============================================================
    # 16. CENTER
    # ============================================================

    M = cv2.moments(
        water_contour
    )

    if M["m00"] != 0:

        center_x = (
            M["m10"] /
            M["m00"]
        )

        center_y = (
            M["m01"] /
            M["m00"]
        )

    else:

        center_x = x + w / 2
        center_y = y + h / 2


    # ============================================================
    # 17. SHAPE ANALYSIS
    # ============================================================

    aspect_ratio = (
        w / h
        if h != 0
        else 0
    )


    # Compactness:
    #
    # 4*pi*Area / Perimeter²
    #
    # Circle ≈ 1
    # Irregular shapes < 1

    compactness = 0

    if perimeter > 0:

        compactness = (
            4 *
            math.pi *
            water_pixel_area /
            (perimeter ** 2)
        )


    # ============================================================
    # 18. CREATE DETECTION IMAGE
    # ============================================================

    detected = rgb.copy()


    # ------------------------------------------------------------
    # Blue transparent overlay
    # ------------------------------------------------------------

    blue_layer = np.zeros_like(
        detected
    )

    blue_layer[:, :] = [
        0,
        100,
        255
    ]


    alpha = 0.45

    water_pixels = (
        final_mask == 255
    )


    detected[water_pixels] = (
        (
            1 - alpha
        ) *
        detected[water_pixels]
        +
        alpha *
        blue_layer[water_pixels]
    ).astype(np.uint8)


    # ============================================================
    # 19. RED WATER BOUNDARY
    # ============================================================

    cv2.drawContours(
        detected,
        [water_contour],
        -1,
        (255, 0, 0),
        4
    )


    # ============================================================
    # 20. YELLOW BOUNDING BOX
    # ============================================================

    cv2.rectangle(
        detected,
        (x, y),
        (x + w, y + h),
        (255, 255, 0),
        3
    )


    # ============================================================
    # 21. CENTER POINT
    # ============================================================

    cv2.circle(
        detected,
        (
            int(center_x),
            int(center_y)
        ),
        8,
        (255, 255, 0),
        -1
    )


    # ============================================================
    # 22. ADD INFORMATION TO IMAGE
    # ============================================================

    font = cv2.FONT_HERSHEY_SIMPLEX


    # Black information box

    box_height = 135

    cv2.rectangle(
        detected,
        (10, 10),
        (390, box_height),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        detected,
        "WATER BODY DETECTED",
        (20, 38),
        font,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        detected,
        f"Coverage: {coverage_percentage:.2f}%",
        (20, 65),
        font,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        detected,
        f"Pixels: {water_mask_pixels}",
        (20, 90),
        font,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        detected,
        f"Center: ({int(center_x)}, {int(center_y)})",
        (20, 115),
        font,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ============================================================
    # 23. DISPLAY ALL RESULTS
    # ============================================================

    plt.figure(
        figsize=(20, 12)
    )


    # ------------------------------------------------------------
    # Original
    # ------------------------------------------------------------

    plt.subplot(2, 3, 1)

    plt.imshow(
        original
    )

    plt.title(
        "1. Original Satellite Image",
        fontsize=14
    )

    plt.axis("off")


    # ------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------

    plt.subplot(2, 3, 2)

    plt.imshow(
        probability,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04
    )

    plt.title(
        "2. Water Probability Map",
        fontsize=14
    )

    plt.axis("off")


    # ------------------------------------------------------------
    # Initial / clean mask
    # ------------------------------------------------------------

    plt.subplot(2, 3, 3)

    plt.imshow(
        initial_mask,
        cmap="gray"
    )

    plt.title(
        "3. Initial Water Segmentation",
        fontsize=14
    )

    plt.axis("off")


    # ------------------------------------------------------------
    # Clean mask
    # ------------------------------------------------------------

    plt.subplot(2, 3, 4)

    plt.imshow(
        final_mask,
        cmap="gray"
    )

    plt.title(
        "4. Clean Water Mask",
        fontsize=14
    )

    plt.axis("off")


    # ------------------------------------------------------------
    # Final detection
    # ------------------------------------------------------------

    plt.subplot(2, 3, 5)

    plt.imshow(
        detected
    )

    plt.title(
        "5. Final Water Body Detection",
        fontsize=14
    )

    plt.axis("off")


    # ------------------------------------------------------------
    # Analysis graph
    # ------------------------------------------------------------

    plt.subplot(2, 3, 6)

    parameters = [
        "Coverage %",
        "Aspect Ratio",
        "Compactness"
    ]

    values = [
        coverage_percentage,
        aspect_ratio * 10,
        compactness * 100
    ]

    plt.bar(
        parameters,
        values
    )

    plt.title(
        "6. Water Body Analysis",
        fontsize=14
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.show()


    # ============================================================
    # 24. PRINT COMPLETE REPORT
    # ============================================================

    print("\n")
    print("=" * 70)
    print("             WATER BODY ANALYSIS REPORT")
    print("=" * 70)

    print("\nIMAGE INFORMATION")
    print("-" * 70)

    print(
        "File:",
        image_path
    )

    print(
        "Width:",
        width,
        "pixels"
    )

    print(
        "Height:",
        height,
        "pixels"
    )

    print(
        "Total pixels:",
        height * width
    )


    print("\nWATER DETECTION")
    print("-" * 70)

    print(
        "Detection status: WATER BODY DETECTED"
    )

    print(
        "Detection threshold:",
        round(threshold_value, 4)
    )


    print("\nWATER AREA")
    print("-" * 70)

    print(
        "Water pixels:",
        int(water_mask_pixels)
    )

    print(
        "Water coverage:",
        round(
            coverage_percentage,
            2
        ),
        "%"
    )


    print("\nBOUNDARY ANALYSIS")
    print("-" * 70)

    print(
        "Perimeter:",
        round(
            perimeter,
            2
        ),
        "pixels"
    )

    print(
        "Bounding box:",
        f"{w} x {h}",
        "pixels"
    )


    print("\nCENTER")
    print("-" * 70)

    print(
        "Center X:",
        round(
            center_x,
            2
        )
    )

    print(
        "Center Y:",
        round(
            center_y,
            2
        )
    )


    print("\nSHAPE ANALYSIS")
    print("-" * 70)

    print(
        "Aspect ratio:",
        round(
            aspect_ratio,
            3
        )
    )

    print(
        "Compactness:",
        round(
            compactness,
            3
        )
    )


    print("\nNOTE")
    print("-" * 70)

    print(
        "Measurements are image-based because the input is JPG/PNG."
    )

    print(
        "Real-world m²/hectares/km² require known image scale."
    )

    print(
        "Exact latitude/longitude requires georeferenced imagery."
    )


    print("\n")
    print("=" * 70)


    # ============================================================
    # 25. SAVE OUTPUT IMAGE
    # ============================================================

    output_image = (
        "water_body_final_detection.png"
    )

    cv2.imwrite(
        output_image,
        cv2.cvtColor(
            detected,
            cv2.COLOR_RGB2BGR
        )
    )


    # ============================================================
    # 26. SAVE MASK
    # ============================================================

    mask_file = (
        "water_body_clean_mask.png"
    )

    cv2.imwrite(
        mask_file,
        final_mask
    )


    # ==
