snvme_devices=$(find /sys/bus/pci/drivers/snvme/ -type l -name "????:??:??.?" -printf "%f\n")

if [ -z "$snvme_devices" ]; then
    echo "No devices found bound to snvme driver."
    exit 0
fi

echo "Found the following snvme devices to be unbound:"
echo "$snvme_devices"

# Execute unbinding
for dev in $snvme_devices; do
    echo -n "$dev" > /sys/bus/pci/drivers/snvme/unbind
    echo "Unbound $dev"
done

echo "All snvme devices have been unbound."