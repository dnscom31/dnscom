package com.example.facemvp;

import android.content.ClipData;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

final class ModelStore {
    static final String ARC = "arcface_w600k_r50.onnx";
    static final String CONVERTER = "crossface_ghost.onnx";
    static final String SWAP = "ghost_1_256.onnx";

    private ModelStore() {}

    static File dir(Context context) {
        File d = new File(context.getFilesDir(), "models");
        if (!d.exists()) d.mkdirs();
        return d;
    }

    static File file(Context context, String name) {
        return new File(dir(context), name);
    }

    static boolean ready(Context context) {
        return valid(file(context, ARC)) && valid(file(context, CONVERTER)) && valid(file(context, SWAP));
    }

    static String status(Context context) {
        List<String> missing = new ArrayList<>();
        if (!valid(file(context, ARC))) missing.add("ArcFace");
        if (!valid(file(context, CONVERTER))) missing.add("CrossFace");
        if (!valid(file(context, SWAP))) missing.add("GHOST");
        return missing.isEmpty() ? "AI 모델 준비 완료" : "필요 모델: " + String.join(", ", missing);
    }

    static int importFromIntent(Context context, Intent data) throws Exception {
        List<Uri> uris = new ArrayList<>();
        if (data.getData() != null) uris.add(data.getData());
        ClipData clip = data.getClipData();
        if (clip != null) {
            for (int i = 0; i < clip.getItemCount(); i++) {
                Uri uri = clip.getItemAt(i).getUri();
                if (uri != null && !uris.contains(uri)) uris.add(uri);
            }
        }

        int copied = 0;
        for (Uri uri : uris) {
            String display = displayName(context, uri);
            String normalized = normalize(display);
            if (normalized == null) continue;
            copy(context, uri, normalized);
            copied++;
        }
        return copied;
    }

    private static String normalize(String name) {
        if (name == null) return null;
        String n = name.toLowerCase(Locale.ROOT);
        if (!n.endsWith(".onnx")) return null;
        if (n.contains("crossface") && n.contains("ghost")) return CONVERTER;
        if (n.contains("ghost") && n.contains("256")) return SWAP;
        if (n.contains("arcface") || n.contains("w600k")) return ARC;
        return null;
    }

    private static void copy(Context context, Uri uri, String name) throws Exception {
        File target = file(context, name);
        File temp = new File(target.getAbsolutePath() + ".tmp");
        ContentResolver cr = context.getContentResolver();
        try (InputStream in = cr.openInputStream(uri);
             FileOutputStream out = new FileOutputStream(temp)) {
            if (in == null) throw new IllegalArgumentException("모델 파일을 열 수 없습니다.");
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) > 0) out.write(buffer, 0, n);
        }
        if (temp.length() < 1024 * 1024) {
            temp.delete();
            throw new IllegalArgumentException(name + " 파일이 너무 작습니다.");
        }
        if (target.exists()) target.delete();
        if (!temp.renameTo(target)) throw new IllegalStateException("모델 파일 저장에 실패했습니다.");
    }

    private static String displayName(Context context, Uri uri) {
        try (Cursor c = context.getContentResolver().query(uri,
                new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                int idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (idx >= 0) return c.getString(idx);
            }
        } catch (Exception ignored) {}
        return uri.getLastPathSegment();
    }

    private static boolean valid(File f) {
        return f.exists() && f.isFile() && f.length() > 1024 * 1024;
    }
}
