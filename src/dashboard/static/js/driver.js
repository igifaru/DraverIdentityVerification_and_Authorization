        async function updateStatus() {
            try {
                const response = await fetch('/api/status/driver');
                const data = await response.json();

                const statusText = document.getElementById('statusText');
                const instructionText = document.getElementById('instructionText');
                const scanLine = document.getElementById('scanningLine');

                statusText.innerText = data.status_display;
                instructionText.innerText = data.instruction;

                // Update styling based on state
                statusText.className = 'status-message';
                if (data.state === 'authorized') {
                    statusText.classList.add('status-authorized');
                    scanLine.style.display = 'none';
                } else if (data.state === 'unauthorized') {
                    statusText.classList.add('status-unauthorized');
                    scanLine.style.display = 'none';
                } else if (data.state === 'scanning') {
                    statusText.classList.add('status-scanning');
                    scanLine.style.display = 'block';
                } else {
                    statusText.classList.add('status-neutral');
                    scanLine.style.display = 'none';
                }

            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }

        // Poll every 500ms for status updates
        setInterval(updateStatus, 500);
        updateStatus(); // Initial call