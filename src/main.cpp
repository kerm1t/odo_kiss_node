#include <iostream>
#include <vector>
#include <Eigen/Core>

// Header direkt einbinden (Pfade werden durch CMake vererbt)
#include <kiss_icp/pipeline/KissICP.hpp>
#include <kiss_icp/core/Threshold.hpp>

int main() {
    // 1. Konfiguration erstellen
    kiss_icp::pipeline::KISSConfig config;
    config.voxel_size = 1.0;
    config.max_range = 60.0;
    config.min_range = 2.0;

    // 2. Pipeline instanziieren
    kiss_icp::pipeline::KissICP odometry(config);

    // 3. Beispiel-Punktwolke generieren (3 Punkte)
    std::vector<Eigen::Vector3d> frame;
    frame.push_back(Eigen::Vector3d(10.0, 0.0, 0.0));
    frame.push_back(Eigen::Vector3d(10.5, 0.2, 0.0));
    frame.push_back(Eigen::Vector3d(11.0, 0.4, 0.1));

    // 4. Zeitstempel (Timestamps) vorbereiten
    // WICHTIG: Die Anzahl der Zeitstempel MUSS exakt mit der Anzahl der Punkte übereinstimmen!
    std::vector<double> timestamps;
    
    // Beispiel A: Punkte haben leicht versetzte Zeiten (typisch für LiDAR-Scans)
    timestamps.push_back(0.00); // Erster Punkt zum Startzeitpunkt t=0
    timestamps.push_back(0.05); // Zweiter Punkt 50ms später
    timestamps.push_back(0.10); // Dritter Punkt 100ms später

/* 
    Hinweis zu Beispiel B: Falls Ihr Sensor keine Einzelzeitstempel liefert, 
    füllen Sie den Vektor einfach komplett mit Nullen (0.0). KISS-ICP überspringt 
    dann das punktweise Deskewing:
    std::vector<double> timestamps(frame.size(), 0.0);
    */

    // 5. Frame INKLUSIVE Zeitstempel in der Registrierung verarbeiten
    // Die Funktion gibt ein Tuple aus (konvertierte Cloud, extrahierte Keypoints) zurück
    auto [registered_frame, keypoints] = odometry.RegisterFrame(frame, timestamps);

    // 6. Berechnete 3D-Transformation (Pose) auslesen
//    Eigen::Matrix4d pose = odometry.pose(); // Fehler!
    Eigen::Matrix4d pose = odometry.pose().matrix();

    
    std::cout << "KISS-ICP erfolgreich mit Timestamps ausgeführt!\n";
    std::cout << "Aktuelle Position (X, Y, Z): " 
              << pose(0,3) << ", " << pose(1,3) << ", " << pose(2,3) << "\n";

    return 0;
}
