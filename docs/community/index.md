# Community

Welcome to the DocETL community! We're excited to have you join us in exploring and improving document extraction and transformation workflows. We are committed to fostering an inclusive community for all people, regardless of technical background.

## Code of Conduct

While we don't have a formal code of conduct page, we expect all community members to treat each other with respect and kindness. We do not tolerate harassment or discrimination of any kind. If you experience any issues, please reach out to the project maintainers immediately.

## Contributions

We welcome contributions from everyone who is interested in improving DocETL. Here's how you can get involved:

1. **Report Issues**: If you encounter a bug or have a feature request, open an issue on our [GitHub repository](https://github.com/ucbepic/docetl/issues).

2. **Join Discussions**: Have a question or want to discuss ideas? Post on our [Discord server](https://discord.gg/fHp7B2X3xx).

3. **Contribute Code**: Look for issues tagged with "help wanted" or "good first issue" on GitHub. These are great starting points for new contributors.

4. **Join Working Groups**: We will create working groups in Discord focused on different project areas as discussed in our [roadmap](roadmap.md). Join the group(s) that interests you most!

To contribute code:

1. Fork the repository on GitHub.
2. Create a new branch for your changes.
3. Make your changes in your branch.
4. Submit a pull request with your changes.

## Connect with Us

- **GitHub Repository**: Contribute to the project or report issues on our [GitHub repo](https://github.com/ucbepic/docetl).
- **Discord Community**: Join our [Discord server](https://discord.gg/fHp7B2X3xx); we're looking to build a vibrant community of people interested in intelligent document processing.
- **Lab Webpages**: We are affiliated with the EPIC Lab at UC Berkeley. Visit our [Lab Page](https://epic.berkeley.edu) for a description of our research. We are also affiliated with the Data Systems and Foundations group at UC Berkeley--visit our [DSF Page](https://dsf.berkeley.edu) for more information.

!!! info "Request a Tutorial or Research Talk"

    Interested in having us give a tutorial or research talk on DocETL? We'd love to connect! Please email shreyashankar@berkeley.edu to set up a time. Let us know what your team is interested in learning about (e.g., tutorial or research) so we can tailor the presentation to your interests.

## Frequently Encountered Issues

### KeyError in Operations

If you're encountering a KeyError, it's often due to missing an unnest operation in your workflow. The unnest operation is crucial for flattening nested data structures.

**Solution**: Add an [unnest operation](../operators/unnest.md) to your pipeline before accessing nested keys. If you're still having trouble, don't hesitate to open an issue on GitHub or ask for help on our Discord server.


### Browser freezing because of stale client storage

Run the following script:
!!! note "Browser Storage Cleanup Script"
    ```js
    // Function to delete all localStorage items with prefix 'docetl_'
    function cleanupDocETLStorage() {
        const prefix = 'docetl_';
        const itemsToDelete = [];
        
        // First, collect all matching keys
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith(prefix)) {
                itemsToDelete.push(key);
            }
        }
        
        // Then delete them and keep count
        const deletedCount = itemsToDelete.length;
        itemsToDelete.forEach(key => {
            localStorage.removeItem(key);
            console.log(`Deleted key: ${key}`);
        });
        
        console.log(`Cleanup complete. Deleted ${deletedCount} items with prefix "${prefix}"`);
    }

    // Execute the cleanup
    cleanupDocETLStorage();
    ```