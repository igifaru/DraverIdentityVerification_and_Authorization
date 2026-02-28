/**
 * driver.js — Driver Verification Terminal
 *
 * Behaviour:
 *  - Polls /api/status/driver every 800 ms
 *  - ALL states except 'unauthorized' → clean camera + guide (no text interruption)
 *  - 'unauthorized' → ACCESS DENIED overlay appears for DENY_HOLD_MS then auto-hides
 *  - event_id deduplication prevents re-showing same denial event
 *  - Live clock in footer
 *  - Status LED colour reflects current state
 */

(function () {
    'use strict';

    /* ── Config ── */
    const POLL_MS = 800;    // status poll interval
    const DENY_HOLD_MS = 6000;   // how long the denial overlay stays visible

    /* ── DOM refs ── */
    const overlay = document.getElementById('alertOverlay');
    const alertSub = document.getElementById('alertSub');
    const statusLed = document.getElementById('statusLed');
    const statusWord = document.getElementById('statusWord');
    const clockEl = document.getElementById('clockDisplay');

    /* ── Internal state ── */
    let _denyTimer = null;
    let _lastEventId = null;

    /* ── Live clock ── */
    function tickClock() {
        const now = new Date();
        clockEl.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
    tickClock();
    setInterval(tickClock, 1000);

    /* ── Show ACCESS DENIED overlay ── */
    function showDenied(message) {
        if (_denyTimer) { clearTimeout(_denyTimer); _denyTimer = null; }

        alertSub.textContent = message || 'Unauthorized driver detected';
        overlay.classList.add('visible');

        // Update header LED → red
        setStatus('denied', 'ACCESS DENIED');

        _denyTimer = setTimeout(hideDenied, DENY_HOLD_MS);
    }

    /* ── Hide overlay, return to scanning state ── */
    function hideDenied() {
        overlay.classList.remove('visible');
        if (_denyTimer) { clearTimeout(_denyTimer); _denyTimer = null; }
        setStatus('scanning', 'SCANNING');
    }

    /* ── Update header LED + label ── */
    function setStatus(state, label) {
        statusLed.className = 'status-led' + (state === 'denied' ? ' denied' : '');
        statusWord.className = 'status-word' + (state === 'denied' ? ' denied' : '');
        statusWord.textContent = label;
    }

    /* ── Poll /api/status/driver ── */
    async function poll() {
        try {
            const res = await fetch('/api/status/driver');
            if (!res.ok) return;
            const data = await res.json();

            if (data.state === 'unauthorized') {
                /* Deduplicate — don't re-show for the same verification event */
                const eid = data.event_id ?? null;
                if (eid !== null && eid === _lastEventId) return;
                _lastEventId = eid;

                showDenied(data.instruction || 'Unauthorized driver detected');

            } else {
                /* Any other state: scanning / authorized / ready / warning
                   → do nothing visually except keep LED correct */
                if (!overlay.classList.contains('visible')) {
                    setStatus('scanning', 'SCANNING');
                }
            }
        } catch (err) {
            /* Network hiccup — camera feed is unaffected */
            console.warn('[driver] poll error:', err.message);
        }
    }

    /* ── Boot ── */
    setInterval(poll, POLL_MS);
    poll();

}());