/* =========================
   CHAT CONFIGURATION
   ========================= */

const API_URL =
    `http://${window.location.hostname}:8080/v1/chat/completions`;

const MODEL =
    "mlx-community/gpt-oss-20b-OptiQ-4bit";


/* =========================
   IMAGE CONFIGURATION
   ========================= */

const IMAGE_GENERATE_API = `http://${window.location.hostname}:8090/generate`;
const IMAGE_EDIT_API = `http://${window.location.hostname}:8090/edit`;


/* =========================
   CHAT STATE
   ========================= */

let messages =
    JSON.parse(
        localStorage.getItem("mlx_conversation") || "[]"
    ).map(message => message.role === "assistant"
        ? {...message, content: extractFinalAnswer(message.content)}
        : message
    ).filter(message => message.role !== "assistant" || message.content);

saveConversation();


const chat =
    document.getElementById("chat");

const input =
    document.getElementById("message");

const send =
    document.getElementById("send");

const newChat =
    document.getElementById("newChat");


/* =========================
   TABS
   ========================= */

const chatTab =
    document.getElementById("chatTab");

const imageTab =
    document.getElementById("imageTab");

const chatView =
    document.getElementById("chatView");

const imageView =
    document.getElementById("imageView");

const editTab = document.getElementById("editTab");
const editView = document.getElementById("editView");


function showChat() {
    chatTab.classList.add("active");
    imageTab.classList.remove("active");
    editTab.classList.remove("active");
    chatView.classList.remove("hidden");
    imageView.classList.add("hidden");
    editView.classList.add("hidden");
    newChat.style.display = "block";
    input.focus();
}

function showImages() {
    chatTab.classList.remove("active");
    imageTab.classList.add("active");
    editTab.classList.remove("active");
    chatView.classList.add("hidden");
    imageView.classList.remove("hidden");
    editView.classList.add("hidden");
    newChat.style.display = "none";
}

function showEdit() {
    chatTab.classList.remove("active");
    imageTab.classList.remove("active");
    editTab.classList.add("active");
    chatView.classList.add("hidden");
    imageView.classList.add("hidden");
    editView.classList.remove("hidden");
    newChat.style.display = "none";
}

/* =========================
   CHAT FUNCTIONS
   ========================= */

function saveConversation() {

    localStorage.setItem(
        "mlx_conversation",
        JSON.stringify(messages)
    );
}


function showMessage(role, text) {

    const div =
        document.createElement("div");

    div.className =
        "message " + role;

    div.textContent = text;

    chat.appendChild(div);

    chat.scrollTop =
        chat.scrollHeight;
}


function redraw() {

    chat.innerHTML = "";

    messages.forEach(message => {

        if (
            message.role === "user" ||
            message.role === "assistant"
        ) {

            showMessage(
                message.role,
                message.content
            );

        }

    });

}


function extractFinalAnswer(content) {

    if (!content) {
        return "";
    }

    /*
    GPT-OSS devuelve:

    <|channel|>analysis
    ...
    <|end|>
    <|start|>assistant
    <|channel|>final
    <|message|>
    RESPUESTA
    */

    const finalChannel =
        /<\|channel\|>final\s*<\|message\|>/;

    if (finalChannel.test(content)) {

        content =
            content.split(finalChannel)[1];

    } else if (content.includes("<|channel|>")) {

        return "";

    }

    content =
        content.replace(
            /<\|(?:end|fim_suffix|im_end)\|>[\s\S]*$/,
            ""
        );

    return content.trim();
}


async function sendMessage() {

    const text =
        input.value.trim();

    if (!text || send.disabled) {
        return;
    }

    input.value = "";

    messages.push({
        role: "user",
        content: text
    });

    showMessage(
        "user",
        text
    );

    saveConversation();

    send.disabled = true;
    send.textContent = "...";

    try {

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            model: MODEL,
                            messages: messages,
                            max_tokens: 500,
                            stream: false
                        })
                }
            );


        if (!response.ok) {

            const detail = await response.text();
            const contentType = response.headers.get("Content-Type") || "";
            const errorMessage = contentType.includes("application/json")
                ? JSON.parse(detail).error || detail
                : detail;

            throw new Error(
                "HTTP " + response.status +
                (errorMessage ? ": " + errorMessage : "")
            );

        }


        const data =
            await response.json();


        console.log(
            "Respuesta MLX:",
            data
        );


        if (
            !data.choices ||
            !data.choices[0] ||
            !data.choices[0].message
        ) {

            throw new Error(
                "Formato de respuesta inesperado"
            );

        }


        const raw =
            data.choices[0].message.content;


        const answer =
            extractFinalAnswer(raw);


        if (!answer) {

            throw new Error(
                data.choices[0].finish_reason === "length"
                    ? "GPT-OSS agotó el límite de tokens antes de responder. " +
                      "Prueba con una pregunta más breve o aumenta max_tokens."
                    : "GPT-OSS no devolvió una respuesta final"
            );

        }


        messages.push({
            role: "assistant",
            content: answer
        });


        showMessage(
            "assistant",
            answer
        );


        saveConversation();

    }

    catch (error) {

        console.error(error);

        showMessage(
            "assistant",
            "❌ Error: " + error.message
        );

    }

    finally {

        send.disabled = false;
        send.textContent = "Enviar";

        input.focus();

    }

}


/* =========================
   IMAGE GENERATION
   ========================= */

const imagePrompt =
    document.getElementById("imagePrompt");

const generateImageButton =
    document.getElementById("generateImage");

const imageStatus =
    document.getElementById("imageStatus");

const generatedImage =
    document.getElementById("generatedImage");


async function generateImage() {

    const prompt =
        imagePrompt.value.trim();

    if (
        !prompt ||
        generateImageButton.disabled
    ) {
        return;
    }


    generateImageButton.disabled = true;
    generateImageButton.textContent =
        "Generando...";

    imageStatus.textContent =
        "⏳ Generando imagen...";

    generatedImage.style.display =
        "none";


    try {

        const response =
            await fetch(
                IMAGE_GENERATE_API,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            prompt: prompt,
                            width: 1024,
                            height: 1024,
                            steps: 4
                        })
                }
            );


        if (!response.ok) {

            const detail = await response.text();
            const contentType = response.headers.get("Content-Type") || "";
            const errorMessage = contentType.includes("application/json")
                ? JSON.parse(detail).error || detail
                : detail;

            throw new Error(
                "HTTP " + response.status +
                (errorMessage ? ": " + errorMessage : "")
            );

        }


        const blob =
            await response.blob();


        const imageUrl =
            URL.createObjectURL(blob);


        generatedImage.src =
            imageUrl;

        generatedImage.style.display =
            "block";


        imageStatus.textContent =
            "✅ Imagen generada.";

    }

    catch (error) {

        console.error(error);

        imageStatus.textContent =
            "❌ Error: " + error.message;

    }

    finally {

        generateImageButton.disabled =
            false;

        generateImageButton.textContent =
            "Generar imagen";

    }

}



/* =========================
   IMAGE EDITING
   ========================= */

const editImageInput = document.getElementById("editImageInput");
const editPrompt = document.getElementById("editPrompt");
const editImageButton = document.getElementById("editImageButton");
const editStatus = document.getElementById("editStatus");
const editPreview = document.getElementById("editPreview");
const editPreviewContainer = document.getElementById("editPreviewContainer");
const editedImage = document.getElementById("editedImage");

editImageInput.addEventListener("change", function () {
    const file = editImageInput.files[0];
    editedImage.style.display = "none";

    if (!file) {
        editPreviewContainer.classList.add("hidden");
        return;
    }

    editPreview.src = URL.createObjectURL(file);
    editPreviewContainer.classList.remove("hidden");
    editStatus.textContent = "Fotografía seleccionada.";
});

async function editImage() {
    const file = editImageInput.files[0];
    const prompt = editPrompt.value.trim();

    if (!file) {
        editStatus.textContent = "Selecciona primero una fotografía.";
        return;
    }

    if (!prompt || editImageButton.disabled) {
        editStatus.textContent = "Escribe los cambios que deseas realizar.";
        return;
    }

    editImageButton.disabled = true;
    editImageButton.textContent = "Editando...";
    editStatus.textContent = "⏳ Editando la fotografía. Puede tardar un poco...";
    editedImage.style.display = "none";

    try {
        const imageData = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(new Error("No se pudo leer la fotografía."));
            reader.readAsDataURL(file);
        });

        const bitmap = await createImageBitmap(file);
        const scale = Math.min(1, 1536 / bitmap.width, 1536 / bitmap.height);
        const width = Math.max(256, Math.round(bitmap.width * scale));
        const height = Math.max(256, Math.round(bitmap.height * scale));
        bitmap.close();

        const response = await fetch(IMAGE_EDIT_API, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                prompt: prompt,
                image: imageData,
                width: width,
                height: height,
                steps: 4
            })
        });

        if (!response.ok) {
            let detail = "";
            try {
                const errorData = await response.json();
                detail = errorData.error || "";
            } catch (_) {}
            throw new Error("HTTP " + response.status +
                (detail ? ": " + detail : ""));
        }

        const blob = await response.blob();
        editedImage.src = URL.createObjectURL(blob);
        editedImage.style.display = "block";
        editStatus.textContent = "✅ Fotografía editada correctamente.";

    } catch (error) {
        console.error(error);
        editStatus.textContent = "❌ Error: " + error.message;
    } finally {
        editImageButton.disabled = false;
        editImageButton.textContent = "Editar imagen";
    }
}

editImageButton.addEventListener("click", editImage);


/* =========================
   EVENT LISTENERS
   ========================= */

send.addEventListener(
    "click",
    sendMessage
);


input.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();

        }

    }
);


newChat.addEventListener(
    "click",
    function() {

        if (
            confirm(
                "¿Borrar esta conversación y empezar una nueva?"
            )
        ) {

            messages = [];

            localStorage.removeItem(
                "mlx_conversation"
            );

            redraw();

        }

    }
);


chatTab.addEventListener(
    "click",
    showChat
);


imageTab.addEventListener(
    "click",
    showImages
);

editTab.addEventListener("click", showEdit);


generateImageButton.addEventListener(
    "click",
    generateImage
);


/* =========================
   INITIAL STATE
   ========================= */

redraw();
