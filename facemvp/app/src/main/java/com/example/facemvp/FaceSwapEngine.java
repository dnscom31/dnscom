package com.example.facemvp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

final class FaceSwapEngine implements AutoCloseable {
    interface Status {
        void onStatus(String message);
    }

    private static final int ARC_SIZE = 112;
    private static final int SWAP_SIZE = 256;

    private final Context context;
    private final Status status;
    private final OrtEnvironment env;
    private final OrtSession arcSession;
    private final OrtSession converterSession;
    private final OrtSession swapSession;
    private final FaceDetector detector;
    private final TemporalTracker tracker = new TemporalTracker();

    private float[] sourceEmbedding;

    FaceSwapEngine(Context context, Status status) throws Exception {
        this.context = context.getApplicationContext();
        this.status = status;
        this.env = OrtEnvironment.getEnvironment();

        status.onStatus("오프라인 얼굴 탐지기 초기화…");
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .enableTracking()
                .build();
        detector = FaceDetection.getClient(options);

        status.onStatus("ArcFace 모델 로딩…");
        arcSession = openSession(copyAsset("models/w600k_r50.onnx"), true);
        status.onStatus("임베딩 변환 모델 로딩…");
        converterSession = openSession(copyAsset("models/crossface_ghost.onnx"), true);
        status.onStatus("얼굴 생성 모델 로딩…");
        swapSession = openSession(copyAsset("models/ghost_1_256.onnx"), true);
    }

    private OrtSession openSession(File model, boolean tryNnapi) throws Exception {
        if (tryNnapi) {
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            try {
                opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                opts.setIntraOpNumThreads(Math.max(2, Math.min(6, Runtime.getRuntime().availableProcessors() - 1)));
                opts.addNnapi();
                OrtSession s = env.createSession(model.getAbsolutePath(), opts);
                opts.close();
                status.onStatus("NNAPI 가속 활성화");
                return s;
            } catch (Throwable ignored) {
                try { opts.close(); } catch (Throwable ignored2) {}
            }
        }

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setIntraOpNumThreads(Math.max(2, Math.min(6, Runtime.getRuntime().availableProcessors() - 1)));
        OrtSession s = env.createSession(model.getAbsolutePath(), opts);
        opts.close();
        status.onStatus("CPU/NEON 호환 모드");
        return s;
    }

    private File copyAsset(String name) throws Exception {
        File dir = new File(context.getFilesDir(), "models");
        if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("Cannot create model directory");
        String base = name.substring(name.lastIndexOf('/') + 1);
        File out = new File(dir, base);
        if (out.exists() && out.length() > 1024 * 1024) return out;

        try (InputStream in = context.getAssets().open(name);
             FileOutputStream fos = new FileOutputStream(out)) {
            byte[] buf = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) fos.write(buf, 0, n);
        }
        return out;
    }

    void prepareSource(Bitmap source) throws Exception {
        status.onStatus("기준 얼굴 분석 중…");
        Face face = largest(detect(source));
        if (face == null) throw new IllegalArgumentException("기준 사진에서 얼굴을 찾지 못했습니다.");

        PointF[] points = landmarks(face);
        if (points == null) throw new IllegalArgumentException("기준 얼굴의 눈·코·입을 안정적으로 찾지 못했습니다.");

        Bitmap aligned = warp(source, MatrixUtils.similarity(points, MatrixUtils.arcFaceTemplate(ARC_SIZE)), ARC_SIZE);
        float[] arcInput = bitmapToNchw(aligned, ARC_SIZE, 127.5f, 127.5f);
        float[] rawEmbedding = runVector(arcSession, arcInput, new long[] {1, 3, ARC_SIZE, ARC_SIZE});
        if (rawEmbedding.length < 512) throw new IllegalStateException("ArcFace embedding 크기가 예상과 다릅니다.");

        float norm = 0f;
        for (float v : rawEmbedding) norm += v * v;
        norm = (float) Math.sqrt(Math.max(norm, 1e-12f));
        for (int i = 0; i < rawEmbedding.length; i++) rawEmbedding[i] /= norm;

        sourceEmbedding = runVector(converterSession, rawEmbedding, new long[] {1, rawEmbedding.length});
        status.onStatus("기준 얼굴 준비 완료");
    }

    Bitmap swapFrame(Bitmap frame, long ptsUs) throws Exception {
        if (sourceEmbedding == null) throw new IllegalStateException("Source face not prepared");

        List<Face> faces = detect(frame);
        Face face = tracker.select(faces);
        if (face == null) {
            tracker.noteMiss();
            return frame;
        }

        Rect box = face.getBoundingBox();
        if (box.width() < 64 || box.height() < 64) {
            tracker.noteMiss();
            return frame;
        }

        float yaw = face.getHeadEulerAngleY();
        float roll = face.getHeadEulerAngleZ();
        if (Math.abs(yaw) > 55f || Math.abs(roll) > 50f) {
            tracker.reset();
            return frame;
        }

        PointF[] raw = landmarks(face);
        if (raw == null) {
            tracker.noteMiss();
            return frame;
        }

        TemporalTracker.Result stabilized = tracker.update(face, raw, ptsUs);
        if (stabilized == null) return frame;

        Matrix cropMatrix = MatrixUtils.similarity(stabilized.points, MatrixUtils.arcFaceTemplate(SWAP_SIZE));
        Bitmap targetCrop = warp(frame, cropMatrix, SWAP_SIZE);
        float[] target = bitmapToNchw(targetCrop, SWAP_SIZE, 0.5f * 255f, 0.5f * 255f);

        Bitmap generated = runSwap(target, targetCrop, yaw, stabilized.motion);
        Bitmap out = frame.copy(Bitmap.Config.ARGB_8888, true);

        Matrix inverse = new Matrix();
        if (!cropMatrix.invert(inverse)) return frame;

        Canvas canvas = new Canvas(out);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(generated, inverse, paint);
        return out;
    }

    private Bitmap runSwap(float[] target, Bitmap targetCrop, float yaw, float motion) throws Exception {
        String sourceName = null;
        String targetName = null;
        Set<String> names = swapSession.getInputNames();
        for (String n : names) {
            String lower = n.toLowerCase();
            if (lower.contains("source")) sourceName = n;
            else if (lower.contains("target")) targetName = n;
        }
        if (sourceName == null || targetName == null) {
            List<String> list = new ArrayList<>(names);
            if (list.size() != 2) throw new IllegalStateException("GHOST 입력 노드 수가 예상과 다릅니다: " + list);
            sourceName = list.get(0);
            targetName = list.get(1);
        }

        try (OnnxTensor srcTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(sourceEmbedding), new long[] {1, sourceEmbedding.length});
             OnnxTensor targetTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(target), new long[] {1, 3, SWAP_SIZE, SWAP_SIZE})) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(sourceName, srcTensor);
            inputs.put(targetName, targetTensor);

            try (OrtSession.Result result = swapSession.run(inputs)) {
                Object v = result.get(0).getValue();
                if (!(v instanceof float[][][][])) {
                    throw new IllegalStateException("GHOST 출력 형식이 예상과 다릅니다: " + v.getClass());
                }
                float[][][][] out = (float[][][][]) v;
                return renderSwap(out[0], targetCrop, yaw, motion);
            }
        }
    }

    private Bitmap renderSwap(float[][][] chw, Bitmap targetCrop, float yaw, float motion) {
        int[] targetPixels = new int[SWAP_SIZE * SWAP_SIZE];
        targetCrop.getPixels(targetPixels, 0, SWAP_SIZE, 0, 0, SWAP_SIZE, SWAP_SIZE);

        double[] gm = new double[3];
        double[] tm = new double[3];
        int count = 0;
        for (int y = 56; y < 208; y += 2) {
            for (int x = 50; x < 206; x += 2) {
                double nx = (x - 128.0) / 78.0;
                double ny = (y - 132.0) / 98.0;
                if (nx * nx + ny * ny > 1.0) continue;
                int p = targetPixels[y * SWAP_SIZE + x];
                tm[0] += (p >> 16) & 255;
                tm[1] += (p >> 8) & 255;
                tm[2] += p & 255;
                gm[0] += to255(chw[0][y][x]);
                gm[1] += to255(chw[1][y][x]);
                gm[2] += to255(chw[2][y][x]);
                count++;
            }
        }
        if (count > 0) {
            for (int c = 0; c < 3; c++) {
                gm[c] /= count;
                tm[c] /= count;
            }
        }

        double[] shift = new double[3];
        for (int c = 0; c < 3; c++) {
            shift[c] = clamp((tm[c] - gm[c]) * 0.65, -28, 28);
        }

        int[] pixels = new int[SWAP_SIZE * SWAP_SIZE];
        double yawScale = Math.max(0.74, 1.0 - Math.abs(yaw) / 190.0);
        double rx = 94.0 * yawScale;
        double ry = 112.0;
        double cx = 128.0;
        double cy = 132.0;
        double feather = 0.18;
        double global = motion > 0.10 ? 0.82 : (Math.abs(yaw) > 38 ? 0.84 : 0.96);

        for (int y = 0; y < SWAP_SIZE; y++) {
            for (int x = 0; x < SWAP_SIZE; x++) {
                int idx = y * SWAP_SIZE + x;
                int r = clamp255((int) Math.round(to255(chw[0][y][x]) + shift[0]));
                int g = clamp255((int) Math.round(to255(chw[1][y][x]) + shift[1]));
                int b = clamp255((int) Math.round(to255(chw[2][y][x]) + shift[2]));

                double dx = (x - cx) / rx;
                double dy = (y - cy) / ry;
                double d = Math.sqrt(dx * dx + dy * dy);
                double alpha;
                if (d >= 1.0) alpha = 0;
                else if (d <= 1.0 - feather) alpha = 1;
                else {
                    double t = (1.0 - d) / feather;
                    alpha = t * t * (3.0 - 2.0 * t);
                }
                alpha *= global;
                int a = clamp255((int) Math.round(alpha * 255.0));
                pixels[idx] = (a << 24) | (r << 16) | (g << 8) | b;
            }
        }

        Bitmap result = Bitmap.createBitmap(SWAP_SIZE, SWAP_SIZE, Bitmap.Config.ARGB_8888);
        result.setPixels(pixels, 0, SWAP_SIZE, 0, 0, SWAP_SIZE, SWAP_SIZE);
        return result;
    }

    private static int to255(float modelValue) {
        return clamp255((int) Math.round((modelValue * 0.5f + 0.5f) * 255f));
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static int clamp255(int v) {
        return Math.max(0, Math.min(255, v));
    }

    private float[] runVector(OrtSession session, float[] input, long[] shape) throws Exception {
        String name = session.getInputNames().iterator().next();
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape);
             OrtSession.Result result = session.run(Collections.singletonMap(name, tensor))) {
            Object value = result.get(0).getValue();
            if (value instanceof float[][]) {
                return ((float[][]) value)[0];
            }
            if (value instanceof float[]) {
                return (float[]) value;
            }
            throw new IllegalStateException("벡터 모델 출력 형식 오류: " + value.getClass());
        }
    }

    private float[] bitmapToNchw(Bitmap bitmap, int size, float mean, float std) {
        Bitmap src = bitmap.getWidth() == size && bitmap.getHeight() == size
                ? bitmap
                : Bitmap.createScaledBitmap(bitmap, size, size, true);
        int[] pixels = new int[size * size];
        src.getPixels(pixels, 0, size, 0, 0, size, size);
        float[] data = new float[3 * size * size];
        int plane = size * size;
        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            float r = (p >> 16) & 255;
            float g = (p >> 8) & 255;
            float b = p & 255;
            data[i] = (r - mean) / std;
            data[plane + i] = (g - mean) / std;
            data[2 * plane + i] = (b - mean) / std;
        }
        return data;
    }

    private Bitmap warp(Bitmap source, Matrix matrix, int size) {
        Bitmap out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(out);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(source, matrix, paint);
        return out;
    }

    private List<Face> detect(Bitmap bitmap) throws Exception {
        return Tasks.await(detector.process(InputImage.fromBitmap(bitmap, 0)));
    }

    private Face largest(List<Face> faces) {
        if (faces == null || faces.isEmpty()) return null;
        return Collections.max(faces, Comparator.comparingInt(f -> f.getBoundingBox().width() * f.getBoundingBox().height()));
    }

    private PointF[] landmarks(Face face) {
        int[] types = {
                FaceLandmark.LEFT_EYE,
                FaceLandmark.RIGHT_EYE,
                FaceLandmark.NOSE_BASE,
                FaceLandmark.MOUTH_LEFT,
                FaceLandmark.MOUTH_RIGHT
        };
        PointF[] pts = new PointF[5];
        for (int i = 0; i < types.length; i++) {
            FaceLandmark lm = face.getLandmark(types[i]);
            if (lm == null) return null;
            PointF p = lm.getPosition();
            pts[i] = new PointF(p.x, p.y);
        }
        return pts;
    }

    @Override
    public void close() throws Exception {
        tracker.reset();
        detector.close();
        arcSession.close();
        converterSession.close();
        swapSession.close();
    }

    private static final class TemporalTracker {
        private PointF[] previous;
        private RectF previousBox;
        private float previousYaw;
        private float previousRoll;
        private Integer trackedId;
        private long previousPts = -1;
        private int misses = 0;

        Face select(List<Face> faces) {
            if (faces == null || faces.isEmpty()) return null;
            if (trackedId != null) {
                for (Face f : faces) {
                    if (trackedId.equals(f.getTrackingId())) return f;
                }
            }
            return Collections.max(faces, Comparator.comparingInt(f -> f.getBoundingBox().width() * f.getBoundingBox().height()));
        }

        Result update(Face face, PointF[] raw, long ptsUs) {
            Rect boxI = face.getBoundingBox();
            RectF box = new RectF(boxI);
            float yaw = face.getHeadEulerAngleY();
            float roll = face.getHeadEulerAngleZ();
            Integer id = face.getTrackingId();

            boolean hardReset = previous == null
                    || previousBox == null
                    || iou(previousBox, box) < 0.25f
                    || Math.abs(yaw - previousYaw) > 18f
                    || Math.abs(roll - previousRoll) > 16f
                    || (previousPts > 0 && ptsUs - previousPts > 180_000)
                    || (trackedId != null && id != null && !trackedId.equals(id));

            float motion = 0f;
            if (!hardReset && previous != null) {
                for (int i = 0; i < raw.length; i++) {
                    float dx = raw[i].x - previous[i].x;
                    float dy = raw[i].y - previous[i].y;
                    motion += (float) Math.sqrt(dx * dx + dy * dy);
                }
                motion = (motion / raw.length) / Math.max(1f, box.width());
                if (motion > 0.18f) hardReset = true;
            }

            PointF[] out = new PointF[raw.length];
            if (hardReset) {
                for (int i = 0; i < raw.length; i++) out[i] = new PointF(raw[i].x, raw[i].y);
                motion = Math.max(motion, 0.12f);
            } else {
                float alpha = Math.max(0.28f, Math.min(0.82f, 0.28f + motion * 3.4f));
                for (int i = 0; i < raw.length; i++) {
                    float x = previous[i].x * (1f - alpha) + raw[i].x * alpha;
                    float y = previous[i].y * (1f - alpha) + raw[i].y * alpha;
                    out[i] = new PointF(x, y);
                }
            }

            previous = out;
            previousBox = box;
            previousYaw = yaw;
            previousRoll = roll;
            trackedId = id;
            previousPts = ptsUs;
            misses = 0;
            return new Result(out, motion, hardReset);
        }

        void noteMiss() {
            misses++;
            if (misses >= 2) reset();
        }

        void reset() {
            previous = null;
            previousBox = null;
            trackedId = null;
            previousPts = -1;
            misses = 0;
        }

        private static float iou(RectF a, RectF b) {
            float l = Math.max(a.left, b.left);
            float t = Math.max(a.top, b.top);
            float r = Math.min(a.right, b.right);
            float bot = Math.min(a.bottom, b.bottom);
            float inter = Math.max(0, r - l) * Math.max(0, bot - t);
            float union = a.width() * a.height() + b.width() * b.height() - inter;
            return union <= 0 ? 0 : inter / union;
        }

        static final class Result {
            final PointF[] points;
            final float motion;
            final boolean reset;
            Result(PointF[] points, float motion, boolean reset) {
                this.points = points;
                this.motion = motion;
                this.reset = reset;
            }
        }
    }
}
