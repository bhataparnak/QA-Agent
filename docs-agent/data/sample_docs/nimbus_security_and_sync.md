# Nimbus Notes: Security, Privacy and Sync Troubleshooting

## Encryption

All notes are encrypted in transit with TLS 1.3 and at rest with AES-256. Pro and Team users can additionally enable end-to-end encryption for individual notebooks. End-to-end encrypted notebooks cannot be recovered if you forget your passphrase, because Nimbus never has access to the key.

## Data location

Customer data is stored in data centers in Frankfurt (EU customers) and Oregon (all other customers). Team plan admins can request EU-only data residency during workspace setup.

## Two-factor authentication

Two-factor authentication (2FA) is available on every plan. Enable it in Settings > Security using an authenticator app or a hardware security key. Team admins can require 2FA for all members of a workspace.

## Sync troubleshooting

If notes are not syncing between devices, try these steps in order:

1. Check that you are signed in to the same account on every device (Settings > Account shows the email).
2. Make sure the app is updated to the latest version. Versions older than 12 months can no longer sync.
3. Pull down on the notes list (mobile) or press Ctrl/Cmd + R (desktop) to force a sync.
4. On mobile, allow background data and disable battery saver for Nimbus.
5. Free accounts can sync only 2 devices. Remove an old device under Settings > Devices.
6. If a note shows a conflict icon, open it and choose which version to keep.

If sync still fails, send a diagnostic report from Help > Send diagnostics and contact support@nimbusnotes.example. Support replies within 24 hours on Pro and within 4 business hours on Team.
