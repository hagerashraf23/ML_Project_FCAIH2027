import gradio as gr

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Cairo"), "sans-serif"],
   radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#070b14",
    body_background_fill_dark="#070b14",
    background_fill_primary="#0f172a",
    background_fill_primary_dark="#0f172a",
    background_fill_secondary="#0b1220",
    background_fill_secondary_dark="#0b1220",

    body_text_color="#f1f5f9",
    body_text_color_dark="#f1f5f9",
    body_text_color_subdued="#7dd3fc",
    body_text_color_subdued_dark="#7dd3fc",

    block_border_color="#1e3a8a",
    block_border_color_dark="#1e3a8a",
    block_title_text_color="#7dd3fc",
    block_title_text_color_dark="#7dd3fc",

    button_primary_background_fill="#a3e635",
    button_primary_background_fill_dark="#a3e635",
    button_primary_background_fill_hover="#bef264",
    button_primary_background_fill_hover_dark="#bef264",
    button_primary_text_color="#070b14",
    button_primary_text_color_dark="#070b14",

    button_secondary_background_fill="#2563eb",
    button_secondary_background_fill_dark="#2563eb",
    button_secondary_background_fill_hover="#3b82f6",
    button_secondary_background_fill_hover_dark="#3b82f6",
    button_secondary_text_color="white",
    button_secondary_text_color_dark="white",
)
css = """
.block, .gr-block, .gradio-container .prose {
    overflow: visible !important;
}
* {
    scrollbar-width: none;
}
*::-webkit-scrollbar {
    display: none;
}
"""
with gr.Blocks(theme=theme, css=css) as demo:

    
    def show_numerical():
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(variant="primary"),
            gr.update(variant="secondary"),
            gr.update(variant="secondary")
        )


    def show_image():
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(variant="secondary"),
            gr.update(variant="primary"),
            gr.update(variant="secondary")
        )


    def show_text():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(variant="secondary"),
            gr.update(variant="secondary"),
            gr.update(variant="primary")
        )

    gr.Markdown(
        "# <span style='color:#a3e635; text-shadow:0 0 12px #a3e635; font-size:1.5em; vertical-align:middle'>⚡︎</span> Sport INFO")
    gr.Markdown(" MACHINE LEARNING PREDICTION APP")
    gr.Markdown("_________________________________________________________________________________________________________________________________________________")
    gr.Markdown("# welcome to the Sport INFO App") 
    gr.Markdown(" This is a simple app to provide information about various sports.")
    gr.Markdown("## Choose your prediction type")
    gr.Markdown("---")

  
    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### MAIN MENU")

            numerical_btn = gr.Button(
                "📊 Numerical",
                variant="secondary"
            )

            image_btn = gr.Button(
                "🖼️ Image",
                variant="primary"
            )

            text_btn = gr.Button(
                "📝 Text",
                variant="secondary"
            )

        with gr.Column(scale=4):

            with gr.Column(visible=False) as numerical_page:
                gr.Markdown("## 📊 Numerical Prediction")

            with gr.Column(visible=True) as image_page:

                gr.Markdown("## 🖼️ Sports Ball Classification")
                gr.Markdown("Upload an image to identify the type of sports ball.")

                gr.Markdown("### Upload Your Image")

                image_input = gr.Image(
                    label="Choose an Image",
                    type="pil"
                )

                predict_btn = gr.Button(
                    "🔍 Predict",
                    variant="primary"
                )

                gr.Markdown("### Prediction Result")

                prediction_output = gr.Label(
                    label="Predicted Class",
                    num_top_classes=3
                )

            with gr.Column(visible=False) as text_page:

                gr.Markdown("## 📝 Sports Text Classification")
                gr.Markdown("Enter your text below to classify it.")

                gr.Markdown("### Enter Your Text")

                text_input = gr.Textbox(
                    label="Your Text",
                    placeholder="Type or paste your text here...",
                    lines=5
                )

                text_predict_btn = gr.Button(
                    "🔍 Predict",
                    variant="primary"
                )

                gr.Markdown("### Prediction Result")

                text_prediction_output = gr.Label(
                    label="Predicted Class"
                )


    numerical_btn.click(
        fn=show_numerical,
        outputs=[
            numerical_page, image_page, text_page,
            numerical_btn, image_btn, text_btn
        ]
    )

    image_btn.click(
        fn=show_image,
        outputs=[
            numerical_page, image_page, text_page,
            numerical_btn, image_btn, text_btn
        ]
    )

    text_btn.click(
        fn=show_text,
        outputs=[
            numerical_page, image_page, text_page,
            numerical_btn, image_btn, text_btn
        ]
    )


demo.launch()