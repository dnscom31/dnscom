package com.example.facemvp;

import android.graphics.Bitmap;
import android.media.MediaCodecInfo;

final class YuvUtils {
    private YuvUtils() {}

    static byte[] bitmapToYuv420(Bitmap bitmap, int colorFormat) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int frame = width * height;
        byte[] out = new byte[frame * 3 / 2];
        int[] pixels = new int[frame];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        boolean semi = colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar;
        int yIndex = 0;
        int uIndex = frame;
        int vIndex = frame + frame / 4;
        int uvIndex = frame;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int p = pixels[y * width + x];
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;

                int yy = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                int uu = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int vv = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

                out[yIndex++] = (byte) clamp(yy);
                if ((y & 1) == 0 && (x & 1) == 0) {
                    if (semi) {
                        out[uvIndex++] = (byte) clamp(uu);
                        out[uvIndex++] = (byte) clamp(vv);
                    } else {
                        out[uIndex++] = (byte) clamp(uu);
                        out[vIndex++] = (byte) clamp(vv);
                    }
                }
            }
        }
        return out;
    }

    private static int clamp(int v) {
        return Math.max(0, Math.min(255, v));
    }
}
