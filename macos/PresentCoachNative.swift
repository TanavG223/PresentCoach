import AppKit
import AVFoundation
import WebKit

private let localOrigin = "http://127.0.0.1:8765"

@main
struct PresentCoachApplication {
    static func main() {
        let application = NSApplication.shared
        let delegate = PresentCoachAppDelegate()
        application.delegate = delegate
        application.setActivationPolicy(.regular)
        application.run()
    }
}

final class PresentCoachAppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var loadingView: NSVisualEffectView!
    private var statusLabel: NSTextField!
    private var retryButton: NSButton!
    private var serverProcess: Process?
    private var logHandle: FileHandle?
    private var isTerminating = false
    private var restartCount = 0

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildApplicationMenu()
        buildWindow()
        connectToLocalServer()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        isTerminating = true
        if let process = serverProcess, process.isRunning {
            process.terminate()
        }
        try? logHandle?.close()
        logHandle = nil
    }

    private func buildApplicationMenu() {
        let mainMenu = NSMenu()
        let applicationItem = NSMenuItem()
        mainMenu.addItem(applicationItem)

        let applicationMenu = NSMenu()
        applicationMenu.addItem(
            withTitle: "About PresentCoach",
            action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
            keyEquivalent: ""
        )
        applicationMenu.addItem(.separator())
        applicationMenu.addItem(
            withTitle: "Hide PresentCoach",
            action: #selector(NSApplication.hide(_:)),
            keyEquivalent: "h"
        )
        applicationMenu.addItem(.separator())
        applicationMenu.addItem(
            withTitle: "Quit PresentCoach",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        applicationItem.submenu = applicationMenu

        let editItem = NSMenuItem()
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu
        mainMenu.addItem(editItem)

        NSApp.mainMenu = mainMenu
    }

    private func buildWindow() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.preferences.isElementFullscreenEnabled = true
        configuration.userContentController.add(self, name: "presentCoachPermissions")
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsMagnification = true

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "PresentCoach"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.minSize = NSSize(width: 960, height: 680)
        window.isReleasedWhenClosed = false
        window.contentView = webView
        window.center()

        loadingView = NSVisualEffectView(frame: .zero)
        loadingView.material = .sidebar
        loadingView.blendingMode = .withinWindow
        loadingView.state = .active
        loadingView.translatesAutoresizingMaskIntoConstraints = false

        let spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.controlSize = .large
        spinner.startAnimation(nil)

        statusLabel = NSTextField(labelWithString: "Starting your private local coach…")
        statusLabel.font = .systemFont(ofSize: 16, weight: .semibold)
        statusLabel.textColor = .labelColor
        statusLabel.alignment = .center

        retryButton = NSButton(title: "Try again", target: self, action: #selector(retryConnection))
        retryButton.bezelStyle = .rounded
        retryButton.isHidden = true

        let stack = NSStackView(views: [spinner, statusLabel, retryButton])
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 14
        stack.translatesAutoresizingMaskIntoConstraints = false
        loadingView.addSubview(stack)
        webView.addSubview(loadingView)

        NSLayoutConstraint.activate([
            loadingView.leadingAnchor.constraint(equalTo: webView.leadingAnchor),
            loadingView.trailingAnchor.constraint(equalTo: webView.trailingAnchor),
            loadingView.topAnchor.constraint(equalTo: webView.topAnchor),
            loadingView.bottomAnchor.constraint(equalTo: webView.bottomAnchor),
            stack.centerXAnchor.constraint(equalTo: loadingView.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: loadingView.centerYAnchor),
        ])

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        guard
            message.name == "presentCoachPermissions",
            message.body as? String == "request",
            webView.url?.scheme == "http",
            webView.url?.host == "127.0.0.1",
            webView.url?.port == 8765
        else { return }

        requestCaptureAccess(for: .video) { [weak self] cameraGranted in
            guard let self else { return }
            self.requestCaptureAccess(for: .audio) { [weak self] microphoneGranted in
                self?.publishCapturePermissions(
                    camera: cameraGranted,
                    microphone: microphoneGranted
                )
            }
        }
    }

    private func requestCaptureAccess(
        for mediaType: AVMediaType,
        completion: @escaping (Bool) -> Void
    ) {
        switch AVCaptureDevice.authorizationStatus(for: mediaType) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: mediaType, completionHandler: completion)
        case .denied, .restricted:
            completion(false)
        @unknown default:
            completion(false)
        }
    }

    private func publishCapturePermissions(camera: Bool, microphone: Bool) {
        let payload = "{\"camera\":\(camera),\"microphone\":\(microphone),\"native\":true}"
        DispatchQueue.main.async { [weak self] in
            self?.webView.evaluateJavaScript(
                "window.dispatchEvent(new CustomEvent('presentcoach-permission-result',{detail:\(payload)}));"
            )
        }
    }

    @objc private func retryConnection() {
        retryButton.isHidden = true
        statusLabel.stringValue = "Starting your private local coach…"
        connectToLocalServer()
    }

    private func connectToLocalServer() {
        checkHealth { [weak self] isReady in
            guard let self else { return }
            if isReady {
                self.loadPrivateAccessURL()
            } else {
                self.launchServer()
            }
        }
    }

    private func checkHealth(completion: @escaping (Bool) -> Void) {
        guard let url = URL(string: "\(localOrigin)/api/health") else {
            completion(false)
            return
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.0
        URLSession.shared.dataTask(with: request) { _, response, _ in
            let ready = (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async { completion(ready) }
        }.resume()
    }

    private func launchServer() {
        guard serverProcess?.isRunning != true else {
            waitForServer(attempt: 0)
            return
        }
        guard
            let rootFile = Bundle.main.url(forResource: "project-root", withExtension: nil),
            let rawRoot = try? String(contentsOf: rootFile, encoding: .utf8),
            !rawRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            showFailure("PresentCoach needs setup. Run scripts/setup_presentcoach.sh once.")
            return
        }
        let projectRoot = rawRoot.trimmingCharacters(in: .whitespacesAndNewlines)
        let pythonURL = URL(fileURLWithPath: projectRoot)
            .appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            showFailure("The local Python environment is missing. Run setup_presentcoach.sh once.")
            return
        }

        let logDirectory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: logDirectory, withIntermediateDirectories: true
        )
        let logURL = logDirectory.appendingPathComponent("PresentCoach.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }

        do {
            let handle = try FileHandle(forWritingTo: logURL)
            try handle.seekToEnd()
            let process = Process()
            process.executableURL = pythonURL
            process.arguments = ["-m", "presentcoach.presentation_server"]
            process.currentDirectoryURL = URL(fileURLWithPath: projectRoot)
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONUNBUFFERED"] = "1"
            process.environment = environment
            process.standardOutput = handle
            process.standardError = handle
            process.terminationHandler = { [weak self] _ in
                DispatchQueue.main.async {
                    guard let self, !self.isTerminating else { return }
                    self.serverProcess = nil
                    try? self.logHandle?.close()
                    self.logHandle = nil
                    if self.restartCount < 1 {
                        self.restartCount += 1
                        self.statusLabel.stringValue = "Recovering the local coach…"
                        self.loadingView.isHidden = false
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                            self.launchServer()
                        }
                    } else {
                        self.showFailure("The local engine stopped. Try again, or inspect PresentCoach.log.")
                    }
                }
            }
            try process.run()
            serverProcess = process
            logHandle = handle
            waitForServer(attempt: 0)
        } catch {
            showFailure("The local engine could not start: \(error.localizedDescription)")
        }
    }

    private func waitForServer(attempt: Int) {
        if attempt >= 60 {
            showFailure("The local engine did not become ready. Try again.")
            return
        }
        checkHealth { [weak self] ready in
            guard let self else { return }
            if ready {
                self.restartCount = 0
                self.loadPrivateAccessURL()
            } else {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                    self.waitForServer(attempt: attempt + 1)
                }
            }
        }
    }

    private func loadPrivateAccessURL() {
        let accessFile = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/PresentCoach/access-url")
        guard
            let rawURL = try? String(contentsOf: accessFile, encoding: .utf8),
            let url = URL(string: rawURL.trimmingCharacters(in: .whitespacesAndNewlines)),
            url.scheme == "http",
            url.host == "127.0.0.1",
            url.port == 8765,
            url.query?.hasPrefix("access=") == true
        else {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
                self?.loadPrivateAccessURL()
            }
            return
        }
        webView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
    }

    private func showFailure(_ message: String) {
        loadingView.isHidden = false
        statusLabel.stringValue = message
        retryButton.isHidden = false
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard webView.url?.host == "127.0.0.1" else { return }
        loadingView.isHidden = true
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let url = navigationAction.request.url else {
            decisionHandler(.cancel)
            return
        }
        let isLocal = url.host == "127.0.0.1" && url.port == 8765
        decisionHandler(isLocal || url.scheme == "blob" ? .allow : .cancel)
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping () -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = "PresentCoach"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping (Bool) -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = "PresentCoach"
        alert.informativeText = message
        alert.addButton(withTitle: "Continue")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: window) { response in
            completionHandler(response == .alertFirstButtonReturn)
        }
    }
}
