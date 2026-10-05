#include "happly.h"
#include <cmath> // isnan()

#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Core>

// KISS-ICP Core & Pipeline
#include <kiss_icp/pipeline/KissICP.hpp>
#include <kiss_icp/core/Threshold.hpp>

// KISS-ICP interner Datei-Helper für PLY/PCD/LAS
#include <kiss_icp/core/Registration.hpp> 
// Hinweis: Falls die Datei-Hilfsfunktionen in Ihrer Version woanders liegen,
// nutzen Sie alternativ eine externe Library wie Tinyply oder Open3D.
// Unten bauen wir zur Sicherheit eine manuelle Funktion ein, falls der interne Reader abweicht.

// Hilfsfunktion: Simuliert Zeitstempel, falls Ihre PLY-Dateien keine pro Punkt enthalten
std::vector<double> createDummyTimestamps(size_t num_points) {
    // Da wir zwei statische Clouds matchen, setzen wir alle Zeitstempel auf 0.0
    // (KISS-ICP überspringt dann das Deskewing)
    return std::vector<double>(num_points, 0.0);
}

int main() {
    // 1. Pfade zu Ihren beiden PLY-Dateien definieren
    std::string ply_file_1 = "cloud_1.ply";
    std::string ply_file_2 = "cloud_2.ply";

    // 2. KISS-ICP Konfiguration & Pipeline initialisieren
    kiss_icp::pipeline::KISSConfig config;
    config.voxel_size = 0.5;  // Präziserer Abgleich für typische File-Scans
////    config.voxel_size = 2.0;  // Präziserer Abgleich für typische File-Scans
    config.max_range = 100.0;
    config.min_range = 0.5;
    
////////    config.voxel_size = 0.05;  // Sehr klein wählen (5 cm), falls Ihre Cloud klein ist
//    config.max_range = 50.0;
    config.min_range = 0.0;    // Min-Range auf 0 setzen!
//    config.voxel_size = 0.f;  // Sehr klein wählen (5 cm), falls Ihre Cloud klein ist
    config.deskew = false;  // single scans, no per-point timestamps

    // WICHTIG FÜR STATISCHE DATEN: Da wir keine kontinuierliche Odometrie-Historie 
    // von einem echten Fahrzeug haben, müssen wir den initialen Suchradius festsetzen.
    // Verhindert das mathematische Wegdriften im ersten Schritt.
    config.initial_threshold = 2.0; 

    kiss_icp::pipeline::KissICP odometry(config);

    // 3. Punktwolken laden
    // HINWEIS: Hier wird vorausgesetzt, dass Sie die Punkte als std::vector<Eigen::Vector3d> vorliegen haben.
    std::vector<Eigen::Vector3d> cloud_1;
    std::vector<Eigen::Vector3d> cloud_2;

    // TODO: Laden Sie hier Ihre PLY-Daten in cloud_1 und cloud_2.
    // Wenn Sie den KISS-ICP-eigenen Datasets-Reader nutzen wollen:
    // cloud_1 = kiss_icp::io::ReadPointCloud(ply_file_1);

    // Beispiel mit happly (Header-only PLY-Reader)
/*    happly::PLYData plyIn1("data/em4_fr_50.ply");
    std::vector<std::vector<double>> raw_pts1 = plyIn1.getVertexPositions();
    for (const auto& p : raw_pts1) {
        cloud_1.push_back(Eigen::Vector3d(p[0], p[1], p[2]));
    }
    happly::PLYData plyIn2("data/em4_fr_50.ply");
    std::vector<std::vector<double>> raw_pts2 = plyIn2.getVertexPositions();
    for (const auto& p : raw_pts2) {
        cloud_2.push_back(Eigen::Vector3d(p[0], p[1], p[2]));
    }
   
    if (cloud_1.empty() || cloud_2.empty()) {
        std::cerr << "Fehler: Punktwolken konnten nicht geladen werden oder sind leer!\n";
        std::cerr << "Bitte stellen Sie sicher, dass cloud_1 und cloud_2 mit Daten befüllt sind.\n";
        // Dummy-Daten einfügen, damit das Code-Beispiel kompilierbar bleibt:
        cloud_1.push_back(Eigen::Vector3d(0, 0, 0)); cloud_1.push_back(Eigen::Vector3d(1, 0, 0));
        cloud_2.push_back(Eigen::Vector3d(0.1, 0, 0)); cloud_2.push_back(Eigen::Vector3d(1.1, 0, 0));
    }
*/
    // 1. Erste Datei einlesen (Nutzen Sie 'auto' für den happly-Typ)
    happly::PLYData plyIn1("/home/wolfgang/git/odo_kiss_node/data/em4_fr_50.ply");
    plyIn1.validate();
//    auto raw_pts1 = plyIn1.getVertexPositions();
    std::vector<std::array<double, 3>> raw_pts1 = plyIn1.getVertexPositions();
    
    // 2. In das von KISS-ICP benötigte Eigen-Format konvertieren
    for (const auto& p : raw_pts1) {
//    for (int j = 0; j < (int)raw_pts1.size(); j++) {
//        if (std::isnan(p[0])) continue;
//        if (std::isnan(raw_pts1[j][0]) || std::isnan(raw_pts1[j][1]) || std::isnan(raw_pts1[j][2])) {
        if (p[0]<0.1f && p[1]<0.1f && p[2]<0.1f) {
            continue;
        }
//        if (j<1000) std::cout << "x:" << raw_pts1[j][0] << "y:" << raw_pts1[j][1] << "z:" << raw_pts1[j][2] << std::endl;
//        cloud_1.push_back(Eigen::Vector3d(raw_pts1[j][0], raw_pts1[j][1], raw_pts1[j][2]));
        double x = static_cast<double>(p[0]);
        double y = static_cast<double>(p[1]);
        double z = static_cast<double>(p[2]);
        cloud_1.push_back(Eigen::Vector3d(x, y, z));
    }
    std::cout << "cloud1:" << cloud_1.size() << std::endl;
//    for (int j = 0; j < 1000; j++) std::cout << "x:" << cloud_1[j][0] << " y:" << cloud_1[j][1] << " z:" << cloud_1[j][2] << std::endl;
    // Debug 2: Wie groß ist die räumliche Ausdehnung (Bounding Box)?
    Eigen::Vector3d min_pt = cloud_1[0];
    Eigen::Vector3d max_pt = cloud_1[0];
    for (const auto& p : cloud_1) {
        min_pt = min_pt.cwiseMin(p);
        max_pt = max_pt.cwiseMax(p);
    }
    std::cout << "Ausdehnung Min: " << min_pt.transpose() << "\n";
    std::cout << "Ausdehnung Max: " << max_pt.transpose() << "\n";


//    if(cloud_1.size() < 100 || cloud_2.size() < 100) {
//        std::cerr << "WARNUNG: Zu wenige Punkte für stabiles ICP!\n";
//    }

    // 3. Zweite Datei einlesen (analog dazu)
    happly::PLYData plyIn2("/home/wolfgang/git/odo_kiss_node/data/em4_fr_51.ply");
    auto raw_pts2 = plyIn2.getVertexPositions();

    for (const auto& p : raw_pts2) {
//        if (std::isnan(p[0])) continue;
//        if (std::isnan(p[0]) || std::isnan(p[1]) || std::isnan(p[2])) {
//            continue;
//        }
        if (p[0]<0.1f && p[1]<0.1f && p[2]<0.1f) continue;
//        cloud_2.push_back(Eigen::Vector3d(p[0], p[1], p[2]));
        double x = static_cast<double>(p[0]);
        double y = static_cast<double>(p[1]);
        double z = static_cast<double>(p[2]);
        cloud_2.push_back(Eigen::Vector3d(x, y, z));
     }
    std::cout << "cloud2:" << cloud_2.size() << std::endl;

//    cloud_1 = kiss_icp::io::ReadPointCloud("data/em4_fr_50.ply");
//    cloud_2 = kiss_icp::io::ReadPointCloud("data/em4_fr_51.ply");

/*
    cloud_1.clear();
    for (double phi = 0; phi < 6.28; phi += 0.1) {
        for (double theta = 0; theta < 3.14; theta += 0.1) {
            double x = 5.0 * sin(theta) * cos(phi);
            double y = 5.0 * sin(theta) * sin(phi);
            double z = 5.0 * cos(theta);
            cloud_1.emplace_back(x, y, z);
        }
    }
*/

    // 4. Erste Punktwolke registrieren (Initialisierung der Karte)
//    std::vector<double> timestamps_1 = createDummyTimestamps(cloud_1.size());
    std::vector<double> timestamps_1(cloud_1.size(), 0.0);
    odometry.RegisterFrame(cloud_1, timestamps_1);

    std::cout << "--- Frame 1 registriert ---\n";

    // Falls Sie eine Initialschätzung (Initial Guess) haben, übergeben Sie diese!
    // Wenn sich das Objekt z.B. ca. 20cm nach vorne bewegt hat:
    Sophus::SE3d initial_guess = Sophus::SE3d(); // Initial als Identität starten
    // initial_guess.translation().x() = 0.2; // Falls grobe Richtung bekannt


    Eigen::Matrix4d pose_after_cloud_1 = odometry.pose().matrix();
//    std::vector<double> timestamps_2 = createDummyTimestamps(cloud_2.size());
    std::vector<double> timestamps_2(cloud_2.size(), 0.1);
    odometry.RegisterFrame(cloud_2, timestamps_2);
///////    odometry.RegisterFrame(cloud_1, timestamps_1);
//    odometry.RegisterFrame(cloud_1, timestamps_2);
    
    // 6. Ergebnis abfragen
    Eigen::Matrix4d pose_after_cloud_2 = odometry.pose().matrix();
    std::cout << "--- Frame 2 registriert ---\n";
    std::cout << "Neue Pose nach Bewegung:\n" << pose_after_cloud_2 << "\n\n";

    // 7. Die reine relative Transformation zwischen Cloud 1 und Cloud 2 berechnen
    // Da pose_after_cloud_1 meist die Einheitsmatrix ist, entspricht pose_after_cloud_2 oft schon direkt der Odometrie.
    Eigen::Matrix4d relative_transform = pose_after_cloud_1.inverse() * pose_after_cloud_2;
    
    std::cout << "Berechnete relative Odometrie (Transformation):\n" << relative_transform << "\n";
    std::cout << "Translation (X, Y, Z): " 
              << relative_transform(0,3) << ", " 
              << relative_transform(1,3) << ", " 
              << relative_transform(2,3) << "\n";

    return 0;
}

