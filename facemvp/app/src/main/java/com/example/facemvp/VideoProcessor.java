package com.example.facemvp;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.media.MediaMuxer;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.Locale;

final class VideoProcessor {
    interface Progress {
        void onProgress(long done, long total, String message);
    }

    private static final int MAX_EDGE = 1280;
    private static final int MAX_FPS = 20;

    private final Context context;
    private final FaceSwapEngine engine;

    VideoProcessor(Context context, FaceSwapEngine engine) {
        this.context = context.getApplicationContext();
        this.engine = engine;
    }

    Uri process(Uri input, Progress progress) throws Exception {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        retriever.setDataSource(context, input);

        long durationMs = parseLong(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION), 0);
        if (durationMs <= 0) throw new IllegalArgumentException("동영상 길이를 읽을 수 없습니다.");

        float sourceFps = parseFloat(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE), MAX_FPS);
        int fps = Math.max(10, Math.min(MAX_FPS, Math.round(sourceFps > 1 ? sourceFps : MAX_FPS)));
        long durationUs = durationMs * 1000L;
        long stepUs = 1_000_000L / fps;
        int totalFrames = Math.max(1, (int) Math.ceil(durationUs / (double) stepUs));

        Bitmap first = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST);
        if (first == null) throw new IllegalArgumentException("첫 동영상 프레임을 읽지 못했습니다.");
        int[] size = fitEven(first.getWidth(), first.getHeight(), MAX_EDGE);
        int outW = size[0];
        int outH = size[1];

        File temp = new File(context.getCacheDir(), "facemvp_video_" + System.currentTimeMillis() + ".mp4");
        AvcEncoder encoder = new AvcEncoder(temp, outW, outH, fps);

        try {
            for (int i = 0; i < totalFrames; i++) {
                long ptsUs = i * stepUs;
                Bitmap raw;
                if (i == 0) {
                    raw = first;
                } else {
                    raw = retriever.getFrameAtTime(Math.min(ptsUs, durationUs - 1), MediaMetadataRetriever.OPTION_CLOSEST);
                }
                if (raw == null) continue;

                Bitmap frame = raw.getWidth() == outW && raw.getHeight() == outH
                        ? raw.copy(Bitmap.Config.ARGB_8888, false)
                        : Bitmap.createScaledBitmap(raw, outW, outH, true);

                Bitmap swapped;
                try {
                    swapped = engine.swapFrame(frame, ptsUs);
                } catch (Throwable frameError) {
                    swapped = frame;
                }

                Bitmap stamped = addWatermark(swapped);
                encoder.queue(stamped, ptsUs);
                progress.onProgress(i + 1L, totalFrames,
                        String.format(Locale.KOREA, "처리 중 %d/%d · %dp", i + 1, totalFrames, outH));

                if (raw != first && !raw.isRecycled()) raw.recycle();
                if (frame != swapped && !frame.isRecycled()) frame.recycle();
                if (swapped != stamped && !swapped.isRecycled()) swapped.recycle();
                if (!stamped.isRecycled()) stamped.recycle();
            }
            encoder.finish();
        } catch (Throwable t) {
            encoder.abort();
            throw t;
        } finally {
            retriever.release();
        }

        progress.onProgress(totalFrames, totalFrames, "원본 오디오 합치는 중…");
        Uri output = remuxWithAudio(temp, input, durationUs);
        //noinspection ResultOfMethodCallIgnored
        temp.delete();
        return output;
    }

    private Bitmap addWatermark(Bitmap source) {
        Bitmap out = source.copy(Bitmap.Config.ARGB_8888, true);
        Canvas c = new Canvas(out);
        Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
        float textSize = Math.max(16f, out.getWidth() * 0.018f);
        text.setTextSize(textSize);
        text.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        String label = "AI face swap";
        float tw = text.measureText(label);
        float pad = textSize * 0.48f;
        float x = out.getWidth() - tw - pad * 1.5f;
        float y = out.getHeight() - pad;

        Paint bg = new Paint(Paint.ANTI_ALIAS_FLAG);
        bg.setColor(0x88000000);
        c.drawRoundRect(x - pad, y - textSize - pad * 0.55f,
                out.getWidth() - pad * 0.5f, y + pad * 0.45f,
                pad * 0.45f, pad * 0.45f, bg);
        text.setColor(0xFFFFFFFF);
        c.drawText(label, x, y, text);
        return out;
    }

    private Uri remuxWithAudio(File encodedVideo, Uri original, long durationUs) throws Exception {
        ContentResolver cr = context.getContentResolver();
        ContentValues values = new ContentValues();
        values.put(MediaStore.Video.Media.DISPLAY_NAME, "FaceMVP_" + System.currentTimeMillis() + ".mp4");
        values.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4");
        values.put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/FaceMVP");
        values.put(MediaStore.Video.Media.IS_PENDING, 1);
        Uri outUri = cr.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values);
        if (outUri == null) throw new IllegalStateException("출력 파일을 만들 수 없습니다.");

        MediaExtractor videoEx = new MediaExtractor();
        MediaExtractor audioEx = new MediaExtractor();
        MediaMuxer muxer = null;
        ParcelFileDescriptor pfd = null;
        boolean ok = false;
        try {
            videoEx.setDataSource(encodedVideo.getAbsolutePath());
            audioEx.setDataSource(context, original, null);

            int videoTrack = findTrack(videoEx, "video/");
            int audioTrack = findTrack(audioEx, "audio/");
            if (videoTrack < 0) throw new IllegalStateException("인코딩된 비디오 트랙이 없습니다.");

            pfd = cr.openFileDescriptor(outUri, "rw");
            if (pfd == null) throw new IllegalStateException("출력 파일을 열 수 없습니다.");
            muxer = new MediaMuxer(pfd.getFileDescriptor(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);

            int muxVideo = muxer.addTrack(videoEx.getTrackFormat(videoTrack));
            int muxAudio = -1;
            if (audioTrack >= 0) muxAudio = muxer.addTrack(audioEx.getTrackFormat(audioTrack));
            muxer.start();

            copyTrack(videoEx, videoTrack, muxer, muxVideo, Long.MAX_VALUE);
            if (audioTrack >= 0) copyTrack(audioEx, audioTrack, muxer, muxAudio, durationUs);
            muxer.stop();
            ok = true;
        } finally {
            if (muxer != null) {
                try { muxer.release(); } catch (Throwable ignored) {}
            }
            try { videoEx.release(); } catch (Throwable ignored) {}
            try { audioEx.release(); } catch (Throwable ignored) {}
            if (pfd != null) try { pfd.close(); } catch (Throwable ignored) {}

            if (ok) {
                ContentValues ready = new ContentValues();
                ready.put(MediaStore.Video.Media.IS_PENDING, 0);
                cr.update(outUri, ready, null, null);
            } else {
                cr.delete(outUri, null, null);
            }
        }
        return outUri;
    }

    private void copyTrack(MediaExtractor ex, int track, MediaMuxer muxer, int muxTrack, long maxPtsUs) {
        ex.selectTrack(track);
        MediaFormat format = ex.getTrackFormat(track);
        int capacity = 1024 * 1024;
        if (format.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
            capacity = Math.max(capacity, format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE));
        }
        ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();

        while (true) {
            buffer.clear();
            int size = ex.readSampleData(buffer, 0);
            if (size < 0) break;
            long pts = ex.getSampleTime();
            if (pts < 0 || pts > maxPtsUs) break;
            info.offset = 0;
            info.size = size;
            info.presentationTimeUs = pts;
            info.flags = ex.getSampleFlags();
            muxer.writeSampleData(muxTrack, buffer, info);
            ex.advance();
        }
        ex.unselectTrack(track);
    }

    private int findTrack(MediaExtractor ex, String prefix) {
        for (int i = 0; i < ex.getTrackCount(); i++) {
            String mime = ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith(prefix)) return i;
        }
        return -1;
    }

    private static int[] fitEven(int w, int h, int maxEdge) {
        float scale = Math.min(1f, maxEdge / (float) Math.max(w, h));
        int ow = Math.max(2, Math.round(w * scale));
        int oh = Math.max(2, Math.round(h * scale));
        if ((ow & 1) == 1) ow--;
        if ((oh & 1) == 1) oh--;
        return new int[] {ow, oh};
    }

    private static long parseLong(String s, long fallback) {
        try { return s == null ? fallback : Long.parseLong(s); }
        catch (Exception e) { return fallback; }
    }

    private static float parseFloat(String s, float fallback) {
        try { return s == null ? fallback : Float.parseFloat(s); }
        catch (Exception e) { return fallback; }
    }

    private static final class AvcEncoder {
        private final MediaCodec codec;
        private final MediaMuxer muxer;
        private final int width;
        private final int height;
        private final int colorFormat;
        private final MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        private int trackIndex = -1;
        private boolean muxerStarted = false;
        private boolean finished = false;

        AvcEncoder(File output, int width, int height, int fps) throws Exception {
            this.width = width;
            this.height = height;

            MediaCodecInfo info = findAvcEncoder();
            if (info == null) throw new IllegalStateException("H.264 인코더를 찾지 못했습니다.");
            colorFormat = chooseColorFormat(info);

            MediaFormat format = MediaFormat.createVideoFormat("video/avc", width, height);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, colorFormat);
            format.setInteger(MediaFormat.KEY_BIT_RATE, Math.max(2_000_000, width * height * 5));
            format.setInteger(MediaFormat.KEY_FRAME_RATE, fps);
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2);

            codec = MediaCodec.createByCodecName(info.getName());
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            codec.start();
            muxer = new MediaMuxer(output.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
        }

        void queue(Bitmap bitmap, long ptsUs) throws Exception {
            Bitmap source = bitmap.getWidth() == width && bitmap.getHeight() == height
                    ? bitmap
                    : Bitmap.createScaledBitmap(bitmap, width, height, true);
            byte[] yuv = YuvUtils.bitmapToYuv420(source, colorFormat);

            int inputIndex;
            do {
                inputIndex = codec.dequeueInputBuffer(10_000);
                if (inputIndex < 0) drain(false);
            } while (inputIndex < 0);

            ByteBuffer in = codec.getInputBuffer(inputIndex);
            if (in == null) throw new IllegalStateException("Encoder input buffer is null");
            in.clear();
            if (in.remaining() < yuv.length) throw new IllegalStateException("Encoder input buffer too small");
            in.put(yuv);
            codec.queueInputBuffer(inputIndex, 0, yuv.length, ptsUs, 0);
            drain(false);

            if (source != bitmap) source.recycle();
        }

        void finish() throws Exception {
            int idx;
            do {
                idx = codec.dequeueInputBuffer(10_000);
                if (idx < 0) drain(false);
            } while (idx < 0);
            codec.queueInputBuffer(idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
            drain(true);
            closeInternal();
            finished = true;
        }

        void abort() {
            try { closeInternal(); } catch (Throwable ignored) {}
        }

        private void drain(boolean eos) {
            while (true) {
                int outIndex = codec.dequeueOutputBuffer(info, eos ? 20_000 : 0);
                if (outIndex == MediaCodec.INFO_TRY_AGAIN_LATER) {
                    if (!eos) return;
                    continue;
                }
                if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    if (muxerStarted) throw new IllegalStateException("Encoder format changed twice");
                    trackIndex = muxer.addTrack(codec.getOutputFormat());
                    muxer.start();
                    muxerStarted = true;
                    continue;
                }
                if (outIndex < 0) continue;

                ByteBuffer out = codec.getOutputBuffer(outIndex);
                if (out != null && (info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0 && info.size > 0) {
                    if (!muxerStarted) throw new IllegalStateException("Muxer not started");
                    out.position(info.offset);
                    out.limit(info.offset + info.size);
                    muxer.writeSampleData(trackIndex, out, info);
                }
                boolean isEos = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                codec.releaseOutputBuffer(outIndex, false);
                if (isEos) return;
                if (!eos) return;
            }
        }

        private void closeInternal() {
            try { codec.stop(); } catch (Throwable ignored) {}
            try { codec.release(); } catch (Throwable ignored) {}
            if (muxerStarted) {
                try { muxer.stop(); } catch (Throwable ignored) {}
            }
            try { muxer.release(); } catch (Throwable ignored) {}
        }

        private static MediaCodecInfo findAvcEncoder() {
            MediaCodecList list = new MediaCodecList(MediaCodecList.ALL_CODECS);
            for (MediaCodecInfo info : list.getCodecInfos()) {
                if (!info.isEncoder()) continue;
                for (String type : info.getSupportedTypes()) {
                    if ("video/avc".equalsIgnoreCase(type)) return info;
                }
            }
            return null;
        }

        private static int chooseColorFormat(MediaCodecInfo info) {
            MediaCodecInfo.CodecCapabilities caps = info.getCapabilitiesForType("video/avc");
            int flexible = -1;
            int semi = -1;
            for (int f : caps.colorFormats) {
                if (f == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar) return f;
                if (f == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible) flexible = f;
                if (f == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar) semi = f;
            }
            if (flexible != -1) return flexible;
            if (semi != -1) return semi;
            throw new IllegalStateException("지원되는 YUV420 인코더 형식이 없습니다.");
        }
    }
}
