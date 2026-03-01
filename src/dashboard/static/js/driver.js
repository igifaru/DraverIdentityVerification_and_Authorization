/**
 * driver.js — Driver Verification Terminal
 *
 * State machine:
 *
 *   IDLE  ──(face detected)──► DETECTING  ──(stable 2s)──► PROCESSING
 *                 ▲                 |                            |
 *                 │          (face lost)                    (result)
 *                 │            returns                      /       \
 *                 └──────────── IDLE          AUTHORIZED    UNAUTHORIZED
 *                                               (3s)  ───────► IDLE
 *                                            UNAUTHORIZED
 *                                               (5s)  ───────► IDLE
 *
 * APIs:
 *   GET  /api/driver/detect   → {face_present, confidence}
 *   POST /api/driver/verify   → {state, driver_name, similarity, event_id}
 */

(function () {
    'use strict';

    /* ── Timing constants ─────────────────────────────────────────── */
    const DETECT_POLL_MS = 800;    // idle poll interval
    const STABLE_MS = 2000;   // face must be present this long before verify
    const AUTH_HOLD_MS = 3000;   // green result shown for
    const DENY_HOLD_MS = 5000;   // red result shown for
    const CLOCK_MS = 1000;   // clock refresh

    /* ── DOM refs ─────────────────────────────────────────────────── */
    const body = document.body;
    const countBar = document.getElementById('countdownBar');
    const authTimer = document.getElementById('authTimer');
    const deniedTimer = document.getElementById('deniedTimer');
    const driverName = document.getElementById('driverName');
    const authMeta = document.getElementById('authMeta');
    const stateLed = document.getElementById('stateLed');
    const stateWord = document.getElementById('stateWord');
    const clockEl = document.getElementById('clockEl');

    /* ── State ────────────────────────────────────────────────────── */
    let state = 'idle';
    let pollTimer = null;
    let stableTimer = null;
    let resetTimer = null;
    let verifying = false;

    /* ── Helpers ──────────────────────────────────────────────────── */
    function setState(s) {
        state = s;
        body.dataset.state = s;
        const labels = {
            idle: 'SCANNING',
            detecting: 'FACE DETECTED',
            processing: 'VERIFYING',
            authorized: 'AUTHORIZED',
            unauthorized: 'ACCESS DENIED',
        };
        stateWord.textContent = labels[s] || s.toUpperCase();
    }

    function clearReset() {
        if (resetTimer) { clearTimeout(resetTimer); resetTimer = null; }
    }

    function clearStable() {
        if (stableTimer) { clearTimeout(stableTimer); stableTimer = null; }
    }

    function stopPoll() {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    }

    /* ── Clock ────────────────────────────────────────────────────── */
    (function tickClock() {
        const now = new Date();
        clockEl.textContent = now.toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
        setTimeout(tickClock, CLOCK_MS);
    }());

    /* ── Animate a timer bar draining from 100% → 0% ── */
    function animateTimer(el, durationMs) {
        el.style.transition = 'none';
        el.style.transform = 'scaleX(1)';
        void el.offsetWidth; // force reflow
        el.style.transition = `transform ${durationMs}ms linear`;
        el.style.transform = 'scaleX(0)';
    }

    /* ── Animate the countdown bar filling 0% → 100% ── */
    function animateCountdown(durationMs) {
        countBar.style.transition = 'none';
        countBar.style.width = '0%';
        void countBar.offsetWidth;
        countBar.style.transition = `width ${durationMs}ms linear`;
        countBar.style.width = '100%';
    }

    /* ── IDLE: poll for a face ──────────────────────────────────── */
    function startIdle() {
        setState('idle');
        verifying = false;
        stopPoll();
        clearStable();
        clearReset();

        pollTimer = setInterval(async () => {
            if (state !== 'idle') return;
            try {
                const res = await fetch('/api/driver/detect');
                const data = await res.json();
                if (data.face_present) {
                    onFaceDetected();
                }
            } catch (e) {
                console.warn('[driver] detect error:', e.message);
            }
        }, DETECT_POLL_MS);
    }

    /* ── DETECTING: stable 2-second countdown ──────────────────── */
    function onFaceDetected() {
        if (state !== 'idle') return;
        setState('detecting');
        stopPoll();
        animateCountdown(STABLE_MS);

        // Keep checking — if face disappears, abort
        let lostCount = 0;
        const checkStable = setInterval(async () => {
            try {
                const res = await fetch('/api/driver/detect');
                const data = await res.json();
                if (!data.face_present) {
                    lostCount++;
                    if (lostCount >= 2) {          // 2 misses in a row = lost
                        clearInterval(checkStable);
                        clearStable();
                        startIdle();               // back to idle
                        return;
                    }
                } else {
                    lostCount = 0;
                }
            } catch (e) { /* ignore */ }
        }, 400);

        // After STABLE_MS → fire full verification
        stableTimer = setTimeout(() => {
            clearInterval(checkStable);
            startVerification();
        }, STABLE_MS);
    }

    /* ── PROCESSING: call /api/driver/verify ───────────────────── */
    async function startVerification() {
        if (verifying) return;
        verifying = true;
        setState('processing');

        try {
            const res = await fetch('/api/driver/verify', { method: 'POST' });
            const data = await res.json();

            if (data.state === 'authorized') {
                showAuthorized(data);
            } else if (data.state === 'unauthorized') {
                showUnauthorized(data);
            } else {
                // no_face — face disappeared during embedding
                startIdle();
            }
        } catch (e) {
            console.error('[driver] verify error:', e);
            startIdle();
        }
    }

    /* ── AUTHORIZED result ──────────────────────────────────────── */
    function showAuthorized(data) {
        setState('authorized');
        driverName.textContent = data.driver_name
            ? `Welcome, ${data.driver_name}`
            : 'Driver Identified';
        authMeta.textContent = data.similarity
            ? `Match confidence: ${(data.similarity * 100).toFixed(1)}%`
            : '';

        animateTimer(authTimer, AUTH_HOLD_MS);
        resetTimer = setTimeout(startIdle, AUTH_HOLD_MS);
    }

    /* ── UNAUTHORIZED result ────────────────────────────────────── */
    function showUnauthorized(data) {
        setState('unauthorized');
        animateTimer(deniedTimer, DENY_HOLD_MS);
        resetTimer = setTimeout(startIdle, DENY_HOLD_MS);
    }

    /* ── Boot ───────────────────────────────────────────────────── */
    if (!document.hidden) {
        console.log('[driver] Tab visible on load, starting polling.');
        startIdle();
    } else {
        console.log('[driver] Tab hidden on load, polling suspended.');
    }

    // Visibility awareness: suspend polling if the tab is hidden
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            stopPoll();
            clearStable();
            console.log('[driver] Tab hidden, polling suspended.');
        } else {
            console.log('[driver] Tab visible, polling resumed.');
            if (state === 'idle' || state === 'detecting' || state === 'processing') {
                startIdle();
            }
        }
    });

}());