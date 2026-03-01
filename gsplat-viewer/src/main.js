import * as SPLAT from "gsplat";

const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");
const overlay = document.getElementById("overlay");

const renderer = new SPLAT.WebGLRenderer();
const scene = new SPLAT.Scene();
const camera = new SPLAT.Camera();
const controls = new SPLAT.OrbitControls(camera, renderer.canvas);

async function loadPLY(url) {
    await SPLAT.PLYLoader.LoadAsync(
        url,
        scene,
        (progress) => {
            progressIndicator.value = progress * 100;
        },
    );
    progressDialog.close();
    overlay.textContent = `FLUOR-GS — ${scene.splatCount ?? "?"} splats loaded`;
}

async function loadFromFile(file) {
    scene.reset();
    progressDialog.showModal();
    progressIndicator.value = 0;

    if (file.name.endsWith(".ply")) {
        await SPLAT.PLYLoader.LoadFromFileAsync(file, scene, (p) => {
            progressIndicator.value = p * 100;
        });
    } else if (file.name.endsWith(".splat")) {
        await SPLAT.Loader.LoadFromFileAsync(file, scene, (p) => {
            progressIndicator.value = p * 100;
        });
    }

    progressDialog.close();
    overlay.textContent = `FLUOR-GS — ${file.name} loaded`;
}

async function main() {
    // Load the pre-trained model from public/
    await loadPLY("/model.ply");

    // Render loop
    const frame = () => {
        controls.update();
        renderer.render(scene, camera);
        requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);

    // Drag-and-drop support for loading other models
    document.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
    });
    document.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer && e.dataTransfer.files.length > 0) {
            loadFromFile(e.dataTransfer.files[0]);
        }
    });
}

main();
