# DPUIPS installation step by step
First Download deb file for host machine from https://developer.nvidia.com/downloads/networking/secure/doca-sdk/doca_2.0.2/doca_202_b83/ubuntu2004/doca-host-repo-ubuntu2004_2.0.2-0.0.7.2.0.2027.1.23.04.0.5.3.0_amd64.deb
Next Steps are
host# sudo dpkg -i doca-host-repo-ubuntu<version>_amd64.deb
host# sudo apt-get update
host# sudo apt install doca-tools
host# sudo apt install -y doca-runtime doca-sdk
host# sudo apt install -y doca-extra
ifconfig tmfifo_net0 192.168.100.1 netmask 255.255.255.252 up
Download DPU bfb image from https://developer.nvidia.com/downloads/networking/secure/doca-sdk/doca_2.0.2/doca_202_b83/doca_2.0.2_bsp_4.0.2_ubuntu_22.04-6.23-04.prod.bfb
  
Create password hash. Run:
host# openssl passwd -1
Password:
Verifying - Password:
$1$3B0RIrfX$TlHry93NFUJzg3Nya00rE1
Add the password hash in quotes to the bf.cfg file:
host# sudo vim bf.cfg
ubuntu_PASSWORD='$1$3B0RIrfX$TlHry93NFUJzg3Nya00rE1'
When running the installation command, use the --config flag to provide the file containing the password:
host# sudo bfb-install --rshim <rshimN> --bfb <image_path.bfb> --config bf.cfg

Above referece installation found at https://docs.nvidia.com/doca/sdk/installation-guide-for-linux/index.html#determining-dpu-device-id
 
# Set Scalabe function for Nvidia Bluefield IPS   
https://docs.nvidia.com/doca/sdk/scalable-functions/index.html
  
# How to run Nvidia Bluefield DOCA IPS
  #Set Hugepages first
  sudo echo 2048 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
  #Genrate DPI rules signature file from suricata based rules file
  
doca_dpi_compiler -i /tmp/ddos.rules -o /tmp/ddos.cdo -f suricata
  #run IPS
  /opt/mellanox/doca/applications/ips/bin/doca_ips -a 0000:03:00.0,class=regex -a auxiliary:mlx5_core.sf.5,sft_en=1 -a auxiliary:mlx5_core.sf.6,sft_en=1 -l 0-7 -- --cdo /tmp/ddos.cdo -p
  
  
  

  
