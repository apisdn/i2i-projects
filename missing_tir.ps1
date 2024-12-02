# Define the path to your folder
$folderPath = ".\gainrangedataset\tir"

# Get a list of all files in the folder
$files = Get-ChildItem -Path $folderPath

# Create a hash table to store the count of each prefix
$prefixCounts = @{}

# Loop through each file and extract the prefix
foreach ($file in $files) {
    # Extract the prefix (assuming the prefix is separated from the suffix by an underscore or any other delimiter)
    $prefix = $file.Name -replace "_.+$", ""
    
    # Increment the count for this prefix
    if ($prefixCounts.ContainsKey($prefix)) {
        $prefixCounts[$prefix]++
    } else {
        $prefixCounts[$prefix] = 1
    }
}

# Find prefixes that do not occur three times
$incorrectPrefixes = $prefixCounts.GetEnumerator() | Where-Object { $_.Value -ne 3 }

# Output the prefixes that do not occur three times
foreach ($prefix in $incorrectPrefixes) {
    Write-Output "Prefix '$($prefix.Key)' occurs $($prefix.Value) times."
}