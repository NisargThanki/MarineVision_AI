
import hashlib

import streamlit as st
import pandas as pd

from PIL import Image

from src.model_loader import load_marine_model
from src.detection import detect_objects
from src.ai_analysis import analyze_with_openrouter
from src.dataset_gallery import load_test_images


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MarineVision AI",
    page_icon="🌊",
    layout="wide"
)


# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_marine_model()


@st.cache_data
def get_test_images():
    return load_test_images()


# --------------------------------------------------
# 3. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🌊 MarineVision AI")

st.sidebar.write(
    "Underwater Marine Object Detection "
    "and Environmental Intelligence"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Image Analysis",
        "Model Information"
    ]
)


# --------------------------------------------------
# 4. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("🌊 MarineVision AI")

    st.subheader(
        "Underwater Marine Object Detection "
        "and Environmental Intelligence System"
    )

    st.write(
        """
        MarineVision AI uses a deep learning object detection model
        to identify selected marine objects and debris in underwater
        images.

        It also integrates an external vision AI service to provide
        additional environmental observations.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Dataset Images",
            "8,610"
        )

    with col2:
        st.metric(
            "Selected Categories",
            "12"
        )

    with col3:
        st.metric(
            "Detection Model",
            "Faster R-CNN"
        )

    st.divider()

    st.subheader("Project Features")

    st.markdown(
        """
        - Underwater object detection
        - Bounding boxes and confidence scores
        - Test dataset image selection
        - Custom image upload
        - AI-powered environmental analysis
        - Model information and evaluation results
        """
    )


# --------------------------------------------------
# 5. IMAGE ANALYSIS
# --------------------------------------------------

elif page == "Image Analysis":

    st.title("🔍 Underwater Image Analysis")

    st.write(
        "Select an image from the test dataset or upload "
        "your own underwater image."
    )

    st.warning(
        "The detection model was trained on 12 selected categories. "
        "Objects outside these categories may be missed or "
        "incorrectly classified."
    )

    st.divider()

    # Choose image source
    image_source = st.radio(
        "Choose image source",
        [
            "Test Dataset",
            "Upload Custom Image"
        ],
        horizontal=True
    )

    image = None
    current_image_key = None
    image_description = ""

    # ----------------------------------------------
    # A. TEST DATASET IMAGE SELECTION
    # ----------------------------------------------

    if image_source == "Test Dataset":

        with st.spinner("Loading test dataset images..."):
            test_images = get_test_images()

        if not test_images:

            st.error("No test images were found.")

        else:

            st.success(
                f"{len(test_images):,} test images available."
            )

            selected_image_label = st.selectbox(
                "Select a test image",
                options=[
                    f"ID {item['image_id']} — {item['file_name']}"
                    for item in test_images
                ]
            )

            selected_image = next(
                item for item in test_images
                if (
                    f"ID {item['image_id']} — {item['file_name']}"
                    == selected_image_label
                )
            )

            image = Image.open(
                selected_image["image_path"]
            ).convert("RGB")

            current_image_key = (
                f"dataset_{selected_image['image_id']}"
            )

            image_description = (
                f"Test Dataset Image: {selected_image['file_name']}"
            )

            st.caption(
                f"Image ID: {selected_image['image_id']}"
            )

    # ----------------------------------------------
    # B. CUSTOM IMAGE UPLOAD
    # ----------------------------------------------

    else:

        uploaded_file = st.file_uploader(
            "Upload an underwater image",
            type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded_file is not None:

            image_bytes = uploaded_file.getvalue()

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            image_hash = hashlib.md5(
                image_bytes
            ).hexdigest()

            current_image_key = f"upload_{image_hash}"

            image_description = (
                f"Uploaded Image: {uploaded_file.name}"
            )

    # ----------------------------------------------
    # C. DISPLAY SELECTED IMAGE
    # ----------------------------------------------

    if image is not None:

        st.divider()

        st.subheader("Selected Image")

        st.image(
            image,
            caption=image_description,
            use_container_width=True
        )

        st.write(
            f"Image dimensions: {image.width} × {image.height}"
        )

        # ------------------------------------------
        # D. OBJECT DETECTION
        # ------------------------------------------

        if st.button(
            "Run Object Detection",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Loading model and detecting marine objects..."
                ):

                    model, device, idx_to_class = get_model()

                    annotated_image, detections = detect_objects(
                        image=image,
                        model=model,
                        device=device,
                        idx_to_class=idx_to_class,
                        confidence_threshold=0.50,
                        iou_threshold=0.30
                    )

                st.session_state["detection_key"] = current_image_key

                st.session_state["annotated_image"] = annotated_image

                st.session_state["detections"] = detections

                st.session_state["ai_analysis_key"] = None

                st.session_state["ai_analysis_result"] = None

                st.success("Object detection completed!")

            except Exception as error:

                st.error(
                    f"Detection failed: {error}"
                )

        # ------------------------------------------
        # E. DISPLAY DETECTION RESULTS
        # ------------------------------------------

        if (
            st.session_state.get("detection_key")
            == current_image_key
        ):

            annotated_image = st.session_state[
                "annotated_image"
            ]

            detections = st.session_state[
                "detections"
            ]

            st.divider()

            st.subheader("Detection Results")

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:

                st.image(
                    annotated_image,
                    caption="Detected Objects",
                    use_container_width=True
                )

            st.divider()

            total_detections = len(detections)

            st.metric(
                "Total Detected Objects",
                total_detections
            )

            if detections:

                st.subheader("Detected Object Details")

                detection_df = pd.DataFrame(detections)

                st.dataframe(
                    detection_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No objects were detected above the "
                    "confidence threshold of 0.50."
                )

            # --------------------------------------
            # F. OPENROUTER AI ANALYSIS
            # --------------------------------------

            st.divider()

            st.subheader(
                "🤖 AI Environmental Analysis"
            )

            st.write(
                """
                Use the external vision AI service to examine
                the image and provide additional environmental
                observations.
                """
            )

            if st.button(
                "Generate AI Environmental Analysis",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "AI is analyzing the underwater image..."
                    ):

                        ai_result = analyze_with_openrouter(
                            image=image,
                            detected_objects=detections
                        )

                    st.session_state[
                        "ai_analysis_result"
                    ] = ai_result

                    st.session_state[
                        "ai_analysis_key"
                    ] = current_image_key

                except Exception as error:

                    st.error(
                        f"AI analysis failed: {error}"
                    )

            if (
                st.session_state.get("ai_analysis_key")
                == current_image_key
                and st.session_state.get("ai_analysis_result")
            ):

                st.markdown(
                    st.session_state["ai_analysis_result"]
                )

                st.caption(
                    "AI-generated observations should be "
                    "reviewed and verified by a human."
                )


# --------------------------------------------------
# 6. MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.title("🧠 Model Information")

    st.subheader("Object Detection Model")

    st.write(
        """
        MarineVision AI uses Faster R-CNN with a ResNet-50
        Feature Pyramid Network (FPN) backbone.
        """
    )

    st.divider()

    st.subheader("Training Configuration")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Training Epochs",
            "22"
        )

        st.metric(
            "Selected Categories",
            "12"
        )

        st.metric(
            "Training Images",
            "5,520"
        )

    with col2:

        st.metric(
            "Validation Images",
            "1,183"
        )

        st.metric(
            "Test Images",
            "1,184"
        )

        st.metric(
            "Framework",
            "PyTorch"
        )

    st.divider()

    st.subheader("Test Evaluation Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("mAP", "0.570")

    with col2:
        st.metric("mAP@50", "0.821")

    with col3:
        st.metric("mAP@75", "0.653")

    with col4:
        st.metric("mAR@100", "0.650")

    st.caption(
        "These are the recorded test evaluation results "
        "from the final project model."
    )

    st.divider()

    st.subheader("Selected Object Categories")

    categories = [
        "animal_etc",
        "animal_shells",
        "animal_sponge",
        "animal_urchin",
        "bag_plastic",
        "bottle_glass",
        "bottle_plastic",
        "can_metal",
        "clothing_fiber",
        "cup_plastic",
        "tire_rubber",
        "tube_cement"
    ]

    for category in categories:

        st.write(f"• {category}")


# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "MarineVision AI | Underwater Object Detection"
)
