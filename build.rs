fn main() {
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=msvcprt");
        println!("cargo:rustc-link-lib=ucrt");
        println!("cargo:rustc-link-lib=vcruntime");
    }
}
