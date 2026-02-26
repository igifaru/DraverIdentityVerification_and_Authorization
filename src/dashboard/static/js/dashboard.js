/**
 * dashboard.js — Draver Control Room
 * Extracted from index.html. No inline code remains in the template.
 */

// ------------------------------------------------------------------
// Utility: non-blocking notification (replaces browser notify())
// ------------------------------------------------------------------
function notify(msg, type = 'err') {
    const c = document.getElementById('toastContainer');
    if (!c) { console.warn('[notify]', msg); return; }
    const el = document.createElement('div');
    el.className = 'toast';
    el.style.cssText = 'min-width:280px;max-width:380px;';
    el.innerHTML = `
        <div class="toast-header" style="align-items:center;">
            <div class="toast-body" style="flex:1;">
                <div class="toast-title" style="color:${type === 'ok' ? 'var(--ok)' : 'var(--err)'}">
                    ${type === 'ok' ? '&#x2713; Success' : '&#x26A0; Error'}
                </div>
                <div class="toast-desc">${msg}</div>
            </div>
            <button class="toast-close" onclick="dismissToast(this.closest('.toast'))">
                <i class="fas fa-times"></i>
            </button>
        </div>`;
    c.appendChild(el);
    setTimeout(() => dismissToast(el), 6000);
}

// ------------------------------------------------------------------
// Theme toggle
// ------------------------------------------------------------------
const THEME_KEY = 'draver_theme';


function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.getElementById('themeIcon').className =
        theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    localStorage.setItem(THEME_KEY, theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
}

// Restore saved preference on load
applyTheme(localStorage.getItem(THEME_KEY) || 'dark');


// ------------------------------------------------------------------
// View switching
// ------------------------------------------------------------------
const VIEW_TITLES = {
    dashboard: 'Operations Center',
    enrollment: 'Biometric Enrollment',
    audit: 'Audit Logs',
    drivers: 'Manage Drivers',
};

function switchView(id) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    document.getElementById('view' + id.charAt(0).toUpperCase() + id.slice(1))
        .classList.add('active');
    document.getElementById('nav-' + id).classList.add('active');
    document.getElementById('viewTitle').textContent = VIEW_TITLES[id] || id;

    if (id === 'audit') fetchAuditLogs();
    if (id === 'drivers') fetchDrivers();
}

// ------------------------------------------------------------------
// Update system status chip
// ------------------------------------------------------------------
function setChip(running) {
    const chip = document.getElementById('sysChip');
    const label = document.getElementById('sysChipLabel');
    if (running) {
        chip.className = 'sys-chip running';
        label.textContent = 'System Running';
    } else {
        chip.className = 'sys-chip stopped';
        label.textContent = 'System Idle';
    }
}

// ------------------------------------------------------------------
// Fetch and render Operations (Enrollment Overview) — real data only
// ------------------------------------------------------------------
async function fetchDashboard() {
    if (!document.getElementById('viewDashboard').classList.contains('active')) return;

    try {
        // Engine status
        const statusRes = await fetch('/api/status');
        if (statusRes.ok) {
            const statusData = await statusRes.json();
            setChip(statusData.system_status === 'active');
        }

        // All drivers data
        const res = await fetch('/api/drivers');
        if (!res.ok) return;
        const drivers = await res.json();

        const total = drivers.length;
        const active = drivers.filter(d => d.status === 'active').length;
        const inactive = total - active;
        const todayStr = new Date().toISOString().split('T')[0];
        const today = drivers.filter(d => d.enrollment_date && d.enrollment_date.startsWith(todayStr)).length;

        // --- Metric cards ---
        document.getElementById('valTotal').textContent = total;
        document.getElementById('valActive').textContent = active;
        document.getElementById('valInactive').textContent = inactive;
        document.getElementById('valToday').textContent = today;

        document.getElementById('subTotal').textContent = total === 1 ? '1 driver registered' : `${total} drivers registered`;
        document.getElementById('subActive').textContent = active === 1 ? '1 driver active' : `${active} drivers active`;
        document.getElementById('subInactive').textContent = inactive > 0 ? `${inactive} suspended/removed` : 'None suspended';
        document.getElementById('subToday').textContent = today > 0 ? `${today} new today` : 'None today';

        // --- Category breakdown ---
        const catCounts = {};
        const catLabels = { B: 'Passenger cars', C: 'Trucks / HGV', D: 'Buses', E: 'Articulated', F: 'Tractors / Agri' };
        drivers.filter(d => d.status === 'active').forEach(d => {
            const cats = Array.isArray(d.categories) ? d.categories : (d.category || 'B').split(',');

            cats.forEach(c => { const k = c.trim(); catCounts[k] = (catCounts[k] || 0) + 1; });
        });
        const catDiv = document.getElementById('categoryBreakdown');
        if (Object.keys(catCounts).length === 0) {
            catDiv.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem;padding:12px 0;">No active drivers enrolled yet.</div>';
        } else {
            const maxCount = Math.max(...Object.values(catCounts));
            catDiv.innerHTML = ['A', 'B', 'C', 'D', 'E'].filter(k => catCounts[k]).map(k => `
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                    <div style="width:28px;height:28px;border-radius:8px;background:var(--accent-subtle);color:var(--accent);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;flex-shrink:0;">${k}</div>
                    <div style="flex:1;">
                        <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px;">
                            <span style="color:var(--text-soft);">${catLabels[k] || k}</span>
                            <span style="font-weight:600;color:var(--text);">${catCounts[k]}</span>
                        </div>
                        <div style="background:var(--bg-raised);border-radius:99px;height:6px;">
                            <div style="background:var(--accent);border-radius:99px;height:6px;width:${(catCounts[k] / maxCount * 100).toFixed(0)}%;transition:width 0.4s;"></div>
                        </div>
                    </div>
                </div>`).join('');
        }

        // --- Phase readiness checklist ---
        const checkItems = [
            { ok: total > 0, label: 'At least one driver enrolled', hint: total > 0 ? `${active} active` : 'Enroll a driver first' },
            { ok: active >= 3, label: '3+ active drivers (recommended)', hint: active >= 3 ? 'Good coverage' : `Need ${3 - active} more` },
            { ok: active > 0, label: 'Biometric embeddings stored', hint: active > 0 ? `${active} embedding(s) ready` : 'Complete enrollment' },
            { ok: true, label: 'Engine in standby mode', hint: 'Camera OFF — awaiting enrollment' },
        ];
        document.getElementById('readinessList').innerHTML = checkItems.map(item => `
            <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:14px;">
                <div style="width:22px;height:22px;border-radius:50%;background:${item.ok ? 'var(--ok-bg)' : 'var(--bg-raised)'};color:${item.ok ? 'var(--ok)' : 'var(--text-muted)'};display:flex;align-items:center;justify-content:center;font-size:0.72rem;flex-shrink:0;margin-top:1px;">
                    <i class="fas ${item.ok ? 'fa-check' : 'fa-minus'}"></i>
                </div>
                <div>
                    <div style="font-size:0.85rem;font-weight:500;color:${item.ok ? 'var(--text)' : 'var(--text-soft)'}">${item.label}</div>
                    <div style="font-size:0.78rem;color:var(--text-muted);margin-top:2px;">${item.hint}</div>
                </div>
            </div>`).join('');

        // --- Recent enrollments table ---
        const tbody = document.getElementById('recentEnrollTbody');
        const countBadge = document.getElementById('enrollCount');
        const recent = [...drivers].sort((a, b) =>
            new Date(b.enrollment_date || 0) - new Date(a.enrollment_date || 0)
        ).slice(0, 10);

        countBadge.textContent = `${total} driver${total !== 1 ? 's' : ''}`;

        if (!recent.length) {
            tbody.innerHTML = '<tr><td colspan="6" class="table-empty">No drivers enrolled yet. Use the Enrollment tab to add drivers.</td></tr>';
        } else {
            tbody.innerHTML = recent.map(d => {
                const enrolled = d.enrollment_date
                    ? new Date(d.enrollment_date).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
                    : '—';
                const statusCls = d.status === 'active' ? 'badge-active' : 'badge-inactive';
                const photoHtml = d.photo_url
                    ? `<img src="${d.photo_url}" alt="" style="width:36px;height:36px;border-radius:6px;object-fit:cover;border:1px solid var(--border-mid);" onerror="this.outerHTML='<i class=\'fas fa-user-circle\' style=\'font-size:1.4rem;color:var(--text-muted)\'></i>'">`
                    : `<i class="fas fa-user-circle" style="font-size:1.4rem;color:var(--text-muted)"></i>`;
                const cats = Array.isArray(d.categories) ? d.categories.join(', ') : (d.category || '—');
                return `<tr>
                    <td>${photoHtml}</td>
                    <td style="font-weight:600;">${d.name}</td>
                    <td style="font-family:var(--font-mono);font-size:0.82rem;">${d.license_number || '—'}</td>
                    <td>${cats}</td>
                    <td style="color:var(--text-muted);font-size:0.82rem;">${enrolled}</td>
                    <td><span class="badge ${statusCls}" style="text-transform:capitalize;">${d.status}</span></td>
                </tr>`;
            }).join('');
        }

    } catch (err) {
        console.error('[fetchDashboard]', err);
    }
}

// ------------------------------------------------------------------
// Audit logs — fetch, filter, delete
// ------------------------------------------------------------------
let _auditAllLogs = [];   // full fetched list for client-side search

async function fetchAuditLogs() {
    const list = document.getElementById('auditList');
    list.innerHTML = '<div class="audit-empty">Loading…</div>';
    const action = document.getElementById('auditActionFilter')?.value || '';
    const url = '/api/audit?limit=200' + (action ? '&action=' + encodeURIComponent(action) : '');
    try {
        const res = await fetch(url);
        _auditAllLogs = await res.json();
        filterAuditLogs();
    } catch (err) {
        list.innerHTML = '<div class="audit-empty">Failed to load audit logs.</div>';
        console.error('[fetchAuditLogs]', err);
    }
}

function filterAuditLogs() {
    const q = (document.getElementById('auditSearch')?.value || '').toLowerCase();
    const data = q
        ? _auditAllLogs.filter(l =>
            (l.action || '').toLowerCase().includes(q) ||
            (l.user_email || '').toLowerCase().includes(q) ||
            (l.details || '').toLowerCase().includes(q) ||
            (l.ip_address || '').toLowerCase().includes(q))
        : _auditAllLogs;

    const list = document.getElementById('auditList');
    const badge = document.getElementById('auditCountBadge');
    const clearBtn = document.getElementById('btnClearAudit');

    badge.textContent = `${data.length} record${data.length !== 1 ? 's' : ''}`;
    clearBtn.disabled = data.length === 0;

    if (!data.length) {
        list.innerHTML = '<div class="audit-empty">No matching audit records found.</div>';
        return;
    }

    const ACTION_ICONS = {
        LOGIN: 'fa-right-to-bracket', LOGOUT: 'fa-right-from-bracket',
        ENROLL_DRIVER: 'fa-user-plus', DELETE_DRIVER: 'fa-user-minus',
        UPDATE_DRIVER: 'fa-user-pen', DELETE_AUDIT_LOG: 'fa-trash',
        CLEAR_AUDIT_LOGS: 'fa-trash-can',
    };

    list.innerHTML = data.map(log => {
        const ts = log.timestamp ? log.timestamp.replace('T', ' ').split('.')[0] : '—';
        let cls = '';
        if (log.action?.includes('FAILURE')) cls = 'err';
        else if (log.action?.includes('ENROLL') || log.action?.includes('START')) cls = 'ok';
        else if (log.action?.includes('DELETE') || log.action?.includes('CLEAR')) cls = 'err';

        const icon = ACTION_ICONS[log.action] || 'fa-circle-dot';

        return `<div class="audit-item ${cls}" id="audit-row-${log.audit_id}">
            <div style="display:flex;align-items:flex-start;gap:12px;flex:1;">
                <div style="width:32px;height:32px;border-radius:8px;background:var(--bg-raised);display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;">
                    <i class="fas ${icon}" style="font-size:0.8rem;color:var(--text-muted);"></i>
                </div>
                <div style="flex:1;">
                    <div class="audit-action">${log.action || '—'}</div>
                    <div class="audit-meta">User: ${log.user_email || '—'} &bull; IP: ${log.ip_address || '—'}</div>
                    <div class="audit-detail">${log.details || ''}</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;flex-shrink:0;">
                <div class="audit-time">${ts}</div>
                <button title="Delete this log entry"
                    style="background:none;border:none;cursor:pointer;color:var(--text-muted);padding:4px 6px;border-radius:6px;transition:color 0.15s,background 0.15s;"
                    onmouseover="this.style.color='var(--err)';this.style.background='var(--err-bg)';"
                    onmouseout="this.style.color='var(--text-muted)';this.style.background='none';"
                    onclick="deleteAuditLog(${log.audit_id})">
                    <i class="fas fa-trash" style="font-size:0.78rem;"></i>
                </button>
            </div>
        </div>`;
    }).join('');
}

async function deleteAuditLog(auditId) {
    if (!confirm('Delete this log entry? This cannot be undone.')) return;
    try {
        const res = await fetch('/api/audit/' + auditId, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            // Remove row with a fade-out animation
            const row = document.getElementById('audit-row-' + auditId);
            if (row) {
                row.style.transition = 'opacity 0.3s, max-height 0.4s';
                row.style.opacity = '0';
                setTimeout(() => row.remove(), 400);
            }
            _auditAllLogs = _auditAllLogs.filter(l => l.audit_id !== auditId);
            // Update badge after removal
            const badge = document.getElementById('auditCountBadge');
            if (badge) badge.textContent = `${_auditAllLogs.length} record${_auditAllLogs.length !== 1 ? 's' : ''}`;
        } else {
            notify('Failed to delete log: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        notify('Network error: ' + e.message);
    }
}

async function clearAuditLogs() {
    const count = _auditAllLogs.length;
    if (count === 0) return;
    if (!confirm(`Clear all ${count} audit log entries? This cannot be undone.`)) return;
    try {
        const res = await fetch('/api/audit', { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            await fetchAuditLogs(); // reload — will show 1 new CLEAR_AUDIT_LOGS entry
        } else {
            notify('Clear failed: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        notify('Network error: ' + e.message);
    }
}

// ------------------------------------------------------------------
// Engine control
// ------------------------------------------------------------------
async function toggleSystem(start) {
    const endpoint = start ? '/api/system/start' : '/api/system/stop';
    try {
        const res = await fetch(endpoint, { method: 'POST' });
        const data = await res.json();
        if (data.success || data.message) fetchDashboard();
    } catch (err) {
        console.error('[toggleSystem]', err);
    }
}

// ------------------------------------------------------------------
// Enrollment workflow — multi-sample capture
// ------------------------------------------------------------------
let enrollmentSamples = [];
const TOTAL_SAMPLES = 5;

async function startEnrollmentWorkflow() {
    const name = document.getElementById('enrollName').value.trim();
    const id = document.getElementById('enrollID').value.trim();
    const msg = document.getElementById('enrollMsg');
    const btn = document.getElementById('btnStart2');

    // Validate
    if (!name || !id) {
        setEnrollMsg('Name and licence number are required.', 'err');
        return;
    }

    btn.disabled = true;
    enrollmentSamples = [];
    setEnrollMsg('Starting camera… please wait.', '');

    const statusText = document.getElementById('captureStatus');
    const progressBar = document.getElementById('captureProgress');
    statusText.style.display = 'block';
    progressBar.style.display = 'block';

    for (let i = 1; i <= TOTAL_SAMPLES; i++) {
        statusText.textContent = `Sample ${i} / ${TOTAL_SAMPLES}`;
        progressBar.style.width = ((i / TOTAL_SAMPLES) * 100) + '%';
        setEnrollMsg(`Capturing sample ${i} of ${TOTAL_SAMPLES} — look directly at the camera…`, '');

        try {
            const res = await fetch('/api/enroll/live', { method: 'POST' });
            const data = await res.json();

            if (data.image) {
                enrollmentSamples.push(data.image);
                const preview = document.getElementById('enrollPreview');
                preview.src = 'data:image/jpeg;base64,' + data.image;
                preview.style.display = 'block';
                document.getElementById('enrollPlaceholder').style.display = 'none';
                setEnrollMsg(`Sample ${i} captured ✓`, 'ok');
            } else {
                // Build a user-friendly hint from the server error
                const err = data.error || 'Capture returned no image';
                let hint = err;
                if (err.toLowerCase().includes('no face')) {
                    hint = 'No face detected — ensure your face is visible and well-lit.';
                } else if (err.toLowerCase().includes('centered')) {
                    hint = 'Face not centered — move closer to the middle of the frame.';
                } else if (err.toLowerCase().includes('small')) {
                    hint = 'Face too small — move closer to the camera.';
                } else if (err.toLowerCase().includes('confidence') || err.toLowerCase().includes('quality')) {
                    hint = 'Face quality too low — improve lighting and face the camera directly.';
                }
                throw new Error(hint);
            }
        } catch (e) {
            setEnrollMsg(`⚠ ${e.message}`, 'err');
            btn.disabled = false;
            await fetch('/api/camera/stop', { method: 'POST' }).catch(() => { });
            statusText.style.display = 'none';
            progressBar.style.width = '0%';
            return;
        }

        await sleep(300);
    }

    statusText.textContent = 'Stopping camera…';
    await fetch('/api/camera/stop', { method: 'POST' }).catch(() => { });

    statusText.textContent = 'Processing…';
    setEnrollMsg('All samples collected. Processing biometric data…', '');
    await commitEnrollment(name, id);
    btn.disabled = false;
}


async function commitEnrollment(name, id) {
    const categories = [...document.querySelectorAll('#categoryGroup input[name="driverCategory"]:checked')]
        .map(cb => cb.value);
    try {
        const res = await fetch('/api/enroll/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, driver_id: id, categories, images: enrollmentSamples }),
        });
        const data = await res.json();

        if (data.success) {
            setEnrollMsg(data.message || 'Driver enrolled successfully.', 'ok');
            setTimeout(resetEnrollForm, 4000);
        } else {
            setEnrollMsg(data.error || 'Enrollment failed.', 'err');
        }
    } catch (e) {
        setEnrollMsg('Network error: ' + e.message, 'err');
    }
}

function setEnrollMsg(text, type) {
    const el = document.getElementById('enrollMsg');
    el.textContent = text;
    el.className = 'enroll-msg' + (type ? ` ${type}` : '');
}

function resetEnrollForm() {
    document.getElementById('enrollName').value = '';
    document.getElementById('enrollID').value = '';
    document.querySelectorAll('#categoryGroup input[name="driverCategory"]')
        .forEach(cb => { cb.checked = cb.value === 'B'; });
    document.getElementById('enrollPreview').style.display = 'none';
    document.getElementById('enrollPlaceholder').style.display = '';
    document.getElementById('captureStatus').style.display = 'none';
    document.getElementById('captureProgress').style.display = 'none';
    document.getElementById('captureProgress').style.width = '0%';
    setEnrollMsg('', '');
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ------------------------------------------------------------------
// Driver Management
// ------------------------------------------------------------------
async function fetchDrivers() {
    try {
        const res = await fetch('/api/drivers');
        const list = await res.json();
        const tbody = document.getElementById('driversTbody');
        document.getElementById('driverCount').textContent =
            list.length + ' driver' + (list.length !== 1 ? 's' : '');

        if (!list.length) {
            tbody.innerHTML = '<tr><td colspan="8" class="table-empty">No drivers enrolled yet.</td></tr>';
            return;
        }

        tbody.innerHTML = list.map((d, i) => {
            const enrolled = d.enrollment_date
                ? new Date(d.enrollment_date).toLocaleDateString()
                : '—';
            const statusCls = d.status === 'active' ? 'badge-active' : 'badge-inactive';
            const photoHtml = d.photo_url
                ? `<img src="${d.photo_url}" alt="" style="width:40px;height:40px;border-radius:6px;object-fit:cover;border:1px solid var(--border-mid);"
                       onerror="this.outerHTML='<i class=fas\\ fa-user-circle style=font-size:1.6rem;color:var(--text-muted)></i>'">`
                : `<i class="fas fa-user-circle" style="font-size:1.6rem;color:var(--text-muted)"></i>`;
            return `<tr>
                <td style="color:var(--text-muted);font-family:var(--font-mono);font-size:0.78rem;">${i + 1}</td>
                <td>${photoHtml}</td>
                <td style="font-weight:600;">${d.name}</td>
                <td style="font-family:var(--font-mono);font-size:0.82rem;">${d.license_number || '—'}</td>
                <td>${d.categories_display || d.category || '—'}</td>
                <td style="color:var(--text-muted);font-size:0.82rem;">${enrolled}</td>
                <td><span class="badge ${statusCls}" style="text-transform:capitalize;">${d.status}</span></td>
                <td style="display:flex;gap:6px;">
                    <button class="btn btn-outline" style="padding:5px 12px;font-size:0.78rem;"
                        onclick="openEditModal(${JSON.stringify(d).replace(/"/g, '&quot;')})">
                        <i class="fas fa-pen"></i> Edit
                    </button>
                    <button class="btn btn-danger" style="padding:5px 12px;font-size:0.78rem;"
                        onclick="deleteDriver(${d.driver_id}, '${d.name.replace(/'/g, "\\'")}')">
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            </tr>`;
        }).join('');
    } catch (e) {
        document.getElementById('driversTbody').innerHTML =
            '<tr><td colspan="8" class="table-empty">Failed to load drivers.</td></tr>';
        console.error('fetchDrivers error:', e);
    }
}

function openEditModal(driver) {
    document.getElementById('editDriverId').value = driver.driver_id;
    document.getElementById('editName').value = driver.name;
    document.getElementById('editLicense').value = driver.license_number || '';
    document.getElementById('editStatus').value = driver.status || 'active';
    // Tick the right category boxes
    const cats = Array.isArray(driver.categories)
        ? driver.categories
        : (driver.category || '').split(',').map(c => c.trim());
    document.querySelectorAll('#editCatGroup input[name="editCat"]').forEach(cb => {
        cb.checked = cats.includes(cb.value);
    });
    document.getElementById('editModal').classList.add('open');
}

function closeEditModal() {
    document.getElementById('editModal').classList.remove('open');
}
// Close on backdrop click
document.getElementById('editModal').addEventListener('click', function (e) {
    if (e.target === this) closeEditModal();
});
// Close on ESC key
document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeEditModal();
});

async function saveDriverEdit() {
    const id = document.getElementById('editDriverId').value;
    const name = document.getElementById('editName').value.trim();
    const license = document.getElementById('editLicense').value.trim();
    const status = document.getElementById('editStatus').value;
    const categories = [...document.querySelectorAll('#editCatGroup input[name="editCat"]:checked')]
        .map(cb => cb.value);

    if (!name) { notify('Name is required.'); return; }
    if (!categories.length) { notify('Select at least one category.'); return; }

    try {
        const res = await fetch('/api/drivers/' + id, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, license_number: license, categories, status }),
        });
        const data = await res.json();
        if (data.success) {
            closeEditModal();
            fetchDrivers();
        } else {
            notify('Update failed: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        notify('Network error: ' + e.message);
    }
}

async function deleteDriver(driverId, driverName) {
    if (!confirm(`Remove "${driverName}" from the system? This will prevent them from being verified.`)) return;
    try {
        const res = await fetch('/api/drivers/' + driverId, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            fetchDrivers();
        } else {
            notify('Delete failed: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        notify('Network error: ' + e.message);
    }
}

// ------------------------------------------------------------------
// Bootstrap
// ------------------------------------------------------------------
// Initialise the alert cursor to "now" so we don't replay old toasts
// on every page load, but still show the alert panel for recent events.
let _alertCursor = Date.now() / 1000;

async function fetchAlerts() {
    try {
        const res = await fetch('/api/alerts?since=' + _alertCursor + '&limit=10');
        const list = await res.json();
        if (!list.length) return;
        // Advance cursor so next poll won't repeat these events
        _alertCursor = Math.max(...list.map(a => a.unix_ts));
        // Show a toast for each new unauthorized event
        list.forEach(a => showToast(a));
    } catch (e) {
        console.error('fetchAlerts error:', e);
    }
}


function showToast(a) {
    const time = new Date(a.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });

    const imgHtml = a.image_url
        ? `<img class="toast-img" src="${a.image_url}" alt="Captured frame"
               onerror="this.outerHTML='<div class=toast-img-placeholder><i class=fas\\ fa-user-slash></i></div>'">`
        : `<div class="toast-img-placeholder"><i class="fas fa-user-slash"></i></div>`;

    const el = document.createElement('div');
    el.className = 'toast';
    el.innerHTML = `
        <div class="toast-header">
            ${imgHtml}
            <div class="toast-body" style="flex:1;">
                <div class="toast-title">&#x26A0; Unauthorized Access Attempt</div>
                <div class="toast-desc">
                    <strong>${a.driver_name}</strong> was not recognized &mdash;
                    similarity <span style="font-family:var(--font-mono);">${a.similarity.toFixed(4)}</span>
                </div>
                <div class="toast-time">${time}</div>
            </div>
            <button class="toast-close" onclick="dismissToast(this.closest('.toast'))">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="toast-vehicle">
            <div class="toast-vehicle-item">
                <span class="toast-vehicle-label">Plate</span>
                <span class="toast-vehicle-value">${a.vehicle_plate || '—'}</span>
            </div>
            <div class="toast-vehicle-item">
                <span class="toast-vehicle-label">Owner</span>
                <span class="toast-vehicle-value">${a.owner_name || '—'}</span>
            </div>
            <div class="toast-vehicle-item">
                <span class="toast-vehicle-label">Liveness</span>
                <span class="toast-vehicle-value" style="color:${a.liveness ? 'var(--ok)' : 'var(--err)'}">
                    ${a.liveness ? 'Pass' : 'Fail'}
                </span>
            </div>
        </div>`;

    document.getElementById('toastContainer').appendChild(el);
    // Auto-dismiss after 10 s
    setTimeout(() => dismissToast(el), 10000);
}

function dismissToast(el) {
    if (!el || el.classList.contains('out')) return;
    el.classList.add('out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
}


fetchDashboard();
setInterval(fetchDashboard, 3000);

fetchAlerts();
setInterval(fetchAlerts, 3000);

