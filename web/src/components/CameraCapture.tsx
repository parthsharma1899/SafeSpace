import { useEffect, useRef, useState } from "react";
import { mediaErrorMessage } from "../media";
import { CheckIcon } from "./Icons";

type State = "idle" | "starting" | "live" | "captured";

export function CameraCapture({ photo, onPhoto }: { photo: Blob | null; onPhoto: (b: Blob | null) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [state, setState] = useState<State>(photo ? "captured" : "idle");
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<string | null>(null);

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setReady(false);
  };

  // Turn the camera off when leaving the screen.
  useEffect(() => () => stopCamera(), []);

  useEffect(() => {
    if (!photo) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(photo);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [photo]);

  const start = async () => {
    setError("");
    onPhoto(null);
    setState("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
      setState("live");
    } catch (e) {
      stopCamera();
      setError(mediaErrorMessage(e, "camera"));
      setState("idle");
    }
  };

  const capture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError("We couldn't take the photo. Please try again.");
          return;
        }
        onPhoto(blob);
        stopCamera();
        setState("captured");
      },
      "image/jpeg",
      0.92,
    );
  };

  const showVideo = state === "starting" || state === "live";

  return (
    <div className="capture">
      <div className={`camera ${showVideo || preview ? "on" : ""}`}>
        <video
          ref={videoRef}
          muted
          playsInline
          onLoadedData={() => setReady(true)}
          style={{ display: showVideo ? "block" : "none" }}
        />
        {state === "captured" && preview && <img src={preview} alt="Your photo" />}
        {showVideo && <div className="face-guide" aria-hidden />}
        {state === "idle" && (
          <div className="camera-empty">
            <p>Face the camera in good light. The photo is analysed and deleted right away.</p>
          </div>
        )}
        {state === "starting" && <div className="camera-status">Starting camera…</div>}
      </div>

      {state === "idle" && (
        <button className="btn secondary" onClick={start}>Turn on camera</button>
      )}
      {state === "live" && (
        <button className="btn secondary" onClick={capture} disabled={!ready}>
          {ready ? "Take photo" : "Getting ready…"}
        </button>
      )}
      {state === "captured" && (
        <div className="captured-row">
          <span className="ok"><CheckIcon /> Photo taken</span>
          <button className="btn link" onClick={start}>Retake</button>
        </div>
      )}
      {error && <p className="field-error">{error}</p>}
    </div>
  );
}
