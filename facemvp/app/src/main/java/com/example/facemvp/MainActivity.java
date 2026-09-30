package com.example.facemvp;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int REQ_SOURCE = 1001;
    private static final int REQ_VIDEO = 1002;

    private Uri sourceUri;
    private Uri videoUri;
    private TextView sourceText;
    private TextView videoText;
    private TextView statusText;
    private ProgressBar progressBar;
    private Button runButton;
    private CheckBox consentCheck;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildUi());
    }

    private View buildUi() {
        int pad = dp(20);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, dp(24), pad, pad);
        root.setGravity(Gravity.CENTER_HORIZONTAL);

        TextView title = new TextView(this);
        title.setText("Face MVP · Offline");
        title.setTextSize(26);
        title.setTextColor(0xFF172033);
        title.setPadding(0, 0, 0, dp(6));
        root.addView(title, matchWrap());

        TextView subtitle = new TextView(this);
        subtitle.setText("사진 1장 + 대상 동영상 → 기기 내부 얼굴 교체\n네트워크 권한 없음 · 결과에 AI face swap 표식");
        subtitle.setTextSize(14);
        subtitle.setTextColor(0xFF5D6678);
        subtitle.setPadding(0, 0, 0, dp(20));
        root.addView(subtitle, matchWrap());

        Button sourceButton = makeButton("1. 얼굴 사진 선택");
        sourceButton.setOnClickListener(v -> pick("image/*", REQ_SOURCE));
        root.addView(sourceButton, matchWrap());

        sourceText = makeInfo("선택 안 됨");
        root.addView(sourceText, matchWrap());

        Button videoButton = makeButton("2. 대상 동영상 선택");
        videoButton.setOnClickListener(v -> pick("video/*", REQ_VIDEO));
        root.addView(videoButton, matchWrap());

        videoText = makeInfo("선택 안 됨");
        root.addView(videoText, matchWrap());

        consentCheck = new CheckBox(this);
        consentCheck.setText("본인 또는 사용 동의를 받은 얼굴/영상입니다.");
        consentCheck.setTextColor(0xFF303849);
        consentCheck.setPadding(0, dp(12), 0, dp(12));
        root.addView(consentCheck, matchWrap());

        runButton = makeButton("3. 얼굴 교체 시작");
        runButton.setEnabled(false);
        runButton.setOnClickListener(v -> runSwap());
        root.addView(runButton, matchWrap());

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(1000);
        progressBar.setProgress(0);
        progressBar.setPadding(0, dp(18), 0, dp(8));
        root.addView(progressBar, new LinearLayout.LayoutParams(-1, dp(42)));

        statusText = makeInfo("대기 중");
        statusText.setTextColor(0xFF3157D5);
        root.addView(statusText, matchWrap());

        TextView note = makeInfo("MVP 기본값: 최대 1280px / 20fps. 급격한 회전·가림에서는 억지로 합성하지 않고 원본 프레임을 유지해 뭉개짐을 줄입니다.");
        note.setPadding(0, dp(14), 0, 0);
        root.addView(note, matchWrap());

        return root;
    }

    private LinearLayout.LayoutParams matchWrap() {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2);
        lp.bottomMargin = dp(8);
        return lp;
    }

    private Button makeButton(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setTextSize(16);
        return b;
    }

    private TextView makeInfo(String text) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setTextSize(13);
        tv.setTextColor(0xFF737B8C);
        return tv;
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    private void pick(String type, int requestCode) {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType(type);
        i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(i, requestCode);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        try {
            getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException ignored) {}

        if (requestCode == REQ_SOURCE) {
            sourceUri = uri;
            sourceText.setText("얼굴 사진: " + shortName(uri));
        } else if (requestCode == REQ_VIDEO) {
            videoUri = uri;
            videoText.setText("대상 동영상: " + shortName(uri));
        }
        runButton.setEnabled(sourceUri != null && videoUri != null);
    }

    private String shortName(Uri uri) {
        String s = uri.getLastPathSegment();
        if (s == null) return "선택됨";
        return s.length() > 42 ? "…" + s.substring(s.length() - 41) : s;
    }

    private void runSwap() {
        if (sourceUri == null || videoUri == null) return;
        if (!consentCheck.isChecked()) {
            Toast.makeText(this, "사용 동의 확인이 필요합니다.", Toast.LENGTH_SHORT).show();
            return;
        }

        runButton.setEnabled(false);
        progressBar.setProgress(0);
        statusText.setText("AI 엔진 초기화 중…");

        worker.execute(() -> {
            FaceSwapEngine engine = null;
            try {
                Bitmap source = loadBitmap(sourceUri);
                engine = new FaceSwapEngine(this, message -> uiStatus(message));
                engine.prepareSource(source);

                VideoProcessor processor = new VideoProcessor(this, engine);
                Uri output = processor.process(videoUri, (done, total, message) -> {
                    int p = total <= 0 ? 0 : (int) Math.min(1000, (done * 1000L) / total);
                    runOnUiThread(() -> {
                        progressBar.setProgress(p);
                        statusText.setText(message);
                    });
                });

                Uri finalOutput = output;
                runOnUiThread(() -> {
                    progressBar.setProgress(1000);
                    statusText.setText("완료: Movies/FaceMVP에 저장됨");
                    Toast.makeText(this, "저장 완료\n" + finalOutput, Toast.LENGTH_LONG).show();
                    runButton.setEnabled(true);
                });
            } catch (Throwable t) {
                String msg = t.getClass().getSimpleName() + ": " + (t.getMessage() == null ? "알 수 없는 오류" : t.getMessage());
                runOnUiThread(() -> {
                    statusText.setText("오류: " + msg);
                    runButton.setEnabled(true);
                });
            } finally {
                if (engine != null) {
                    try { engine.close(); } catch (Exception ignored) {}
                }
            }
        });
    }

    private void uiStatus(String message) {
        runOnUiThread(() -> statusText.setText(message));
    }

    private Bitmap loadBitmap(Uri uri) throws IOException {
        if (Build.VERSION.SDK_INT >= 28) {
            ImageDecoder.Source src = ImageDecoder.createSource(getContentResolver(), uri);
            return ImageDecoder.decodeBitmap(src, (decoder, info, source) -> {
                decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                decoder.setMemorySizePolicy(ImageDecoder.MEMORY_POLICY_LOW_RAM);
            }).copy(Bitmap.Config.ARGB_8888, false);
        }
        @SuppressWarnings("deprecation")
        Bitmap b = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
        return b.copy(Bitmap.Config.ARGB_8888, false);
    }

    @Override
    protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }
}
