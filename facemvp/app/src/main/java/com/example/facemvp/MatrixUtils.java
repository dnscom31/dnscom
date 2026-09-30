package com.example.facemvp;

import android.graphics.Matrix;
import android.graphics.PointF;

final class MatrixUtils {
    private MatrixUtils() {}

    static Matrix similarity(PointF[] src, PointF[] dst) {
        if (src.length != dst.length || src.length < 2) {
            throw new IllegalArgumentException("Point count mismatch");
        }

        double sx = 0, sy = 0, dx = 0, dy = 0;
        for (int i = 0; i < src.length; i++) {
            sx += src[i].x;
            sy += src[i].y;
            dx += dst[i].x;
            dy += dst[i].y;
        }
        sx /= src.length;
        sy /= src.length;
        dx /= dst.length;
        dy /= dst.length;

        double den = 0;
        double numa = 0;
        double numb = 0;
        for (int i = 0; i < src.length; i++) {
            double x = src[i].x - sx;
            double y = src[i].y - sy;
            double u = dst[i].x - dx;
            double v = dst[i].y - dy;
            den += x * x + y * y;
            numa += x * u + y * v;
            numb += x * v - y * u;
        }
        if (den < 1e-9) throw new IllegalArgumentException("Degenerate landmarks");

        float a = (float) (numa / den);
        float b = (float) (numb / den);
        float tx = (float) (dx - a * sx + b * sy);
        float ty = (float) (dy - b * sx - a * sy);

        Matrix m = new Matrix();
        m.setValues(new float[] {
                a, -b, tx,
                b,  a, ty,
                0,  0,  1
        });
        return m;
    }

    static PointF[] arcFaceTemplate(int size) {
        float s = size / 112.0f;
        return new PointF[] {
                new PointF(38.2946f * s, 51.6963f * s),
                new PointF(73.5318f * s, 51.5014f * s),
                new PointF(56.0252f * s, 71.7366f * s),
                new PointF(41.5493f * s, 92.3655f * s),
                new PointF(70.7299f * s, 92.2041f * s)
        };
    }
}
