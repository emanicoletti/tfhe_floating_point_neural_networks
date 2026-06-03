use std::fs::{self, File};
use std::io::{self};
use std::path::Path;

/// Ensures the required dataset exists; downloads it from Zenodo if missing.
pub fn ensure_dataset_exists(
    res_path: &str,
    zenodo_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(res_path);

    // If the folder/file already exists, do nothing and proceed!
    if path.exists() {
        return Ok(());
    }

    println!("Dataset not found locally at `{}`.", res_path);
    println!("Downloading preprocessed asset from Zenodo archive...");
    println!("Please wait, this might take a moment depending on your network connection...");

    // 1. Create assets directory if missing
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // 2. Download the zip file from Zenodo
    let zip_tmp_path = "res/temp_dataset.zip";
    let response = reqwest::blocking::get(zenodo_url)?;

    // --- NEW: Guard rail to catch bad URLs or Zenodo server errors ---
    if !response.status().is_success() {
        return Err(format!(
            "Failed to download dataset. Zenodo responded with status: {}",
            response.status()
        )
        .into());
    }

    let mut dest = File::create(zip_tmp_path)?;
    // Ensure we stream the body text correctly
    let mut response_body = response;
    io::copy(&mut response_body, &mut dest)?;

    println!("Download complete. Extracting files...");

    // 3. Extract the downloaded zip file
    let zip_file = File::open(zip_tmp_path)?;
    let mut archive = zip::ZipArchive::new(zip_file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = match file.enclosed_name() {
            Some(path) => Path::new("res/").join(path),
            None => continue,
        };

        if (*file.name()).ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }

    // 4. Clean up the temporary zip file
    fs::remove_file(zip_tmp_path)?;
    println!("Dataset ready and verified!\n");

    Ok(())
}
