async function initializeSwagger() {
    const status = document.getElementById("swagger-status");
    try {
        if (!window.SwaggerUIBundle) {
            throw new Error("No se pudo cargar Swagger UI desde el CDN.");
        }
        const response = await fetch("/openapi.json");
        if (!response.ok) {
            throw new Error(`No se pudo cargar OpenAPI (HTTP ${response.status}).`);
        }
        const spec = await response.json();
        for (const path of Object.values(spec.paths)) {
            for (const operation of Object.values(path)) {
                for (const server of operation.servers || []) {
                    if (server.variables && server.variables.hostname) {
                        server.variables.hostname.default = window.location.hostname;
                    }
                }
            }
        }
        window.ui = SwaggerUIBundle({
            spec,
            dom_id: "#swagger-ui",
            deepLinking: true,
            validatorUrl: null,
            persistAuthorization: false,
        });
    } catch (error) {
        console.error("Swagger initialization failed:", error);
        status.textContent = error.message;
    }
}

initializeSwagger();
