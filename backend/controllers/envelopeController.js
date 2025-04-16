let envelopes = []  // This will temporarily store envelopes in memory

const createEnvelope = (req,res,next) => {
    const {name , amount} = req.body;

    if (!name || typeof name !== 'string' || !amount || typeof amount !== 'number' || amount <= 0) {
        return res.status(400).json({ error: 'Invalid envelope name or amount.' });
    }

    // Create the envelope object
    const newEnvelope = {
        id: envelopes.length + 1,  // Generate an ID
        name: name,
        amount: amount
    };

    // Add the new envelope to the array
    envelopes.push(newEnvelope);

    // Respond with the created envelope
    return res.status(201).json(newEnvelope);
};

const getAllEnvelopes = (req,res,next) => {
    res.status(200).json(envelopes)
}

const getEnvelopeById = (req, res) => {
    const { id } = req.params; // Get ID from request params
    const envelope = envelopes.find(env => env.id === parseInt(id)); // Find the envelope with the matching ID

    if (!envelope) {
        return res.status(404).json({ message: 'Envelope not found' }); // If no envelope is found, return a 404
    }

    res.status(200).json(envelope); // If envelope is found, return it with a 200 status
};

// Controller for updating an envelope by ID
const updateEnvelope = (req, res) => {
    const { id } = req.params;  // Get the ID from the request params
    const { name, amount } = req.body;  // Get the new data from the request body
    const envelope = envelopes.find(env => env.id === parseInt(id));  // Find the envelope by its ID

    if (!envelope) {
        return res.status(404).json({ message: 'Envelope not found' });  // If not found, return a 404 error
    }

    // Update the envelope properties if they are provided in the request body
    if (name) envelope.name = name;
    if (amount) {
        if (amount < 0) {
            return res.status(400).json({ message: 'Amount cannot be negative' });
        }
        envelope.amount = amount;
    }

    // Return the updated envelope
    res.status(200).json(envelope);
};

// Controller for deleting an envelope by ID
const deleteEnvelope = (req, res) => {
    const { id } = req.params;  // Get the ID from the request params
    const envelopeIndex = envelopes.findIndex(env => env.id === parseInt(id));  // Find the index of the envelope

    if (envelopeIndex === -1) {
        return res.status(404).json({ message: 'Envelope not found' });  // If not found, return a 404 error
    }

    // Remove the envelope from the array
    envelopes.splice(envelopeIndex, 1);

    // Respond with a 204 No Content status, indicating successful deletion
    res.status(204).send();
};

// Controller to transfer an amount from one envelope to another
const transferAmount = (req, res) => {
    const { fromId, toId } = req.params;  // Source and destination envelope IDs
    const { amount } = req.body;  // Amount to transfer

    // Find the source and destination envelopes
    const sourceEnvelope = envelopes.find(env => env.id === parseInt(fromId));
    const destinationEnvelope = envelopes.find(env => env.id === parseInt(toId));

    // Validate the envelopes and amount
    if (!sourceEnvelope || !destinationEnvelope) {
        return res.status(404).json({ error: 'One or both envelopes not found' });
    }
    if (amount <= 0 || typeof amount !== 'number') {
        return res.status(400).json({ error: 'Transfer amount must be a positive number' });
    }
    if (sourceEnvelope.amount < amount) {
        return res.status(400).json({ error: 'Insufficient funds in the source envelope' });
    }

    // Transfer the amount
    sourceEnvelope.amount -= amount;
    destinationEnvelope.amount += amount;

    res.status(200).json({
        message: `Transferred ${amount} from envelope ${fromId} to envelope ${toId}`,
        sourceEnvelope,
        destinationEnvelope
    });
};

module.exports = {
    createEnvelope,
    getAllEnvelopes,
    getEnvelopeById,
    updateEnvelope,
    deleteEnvelope,
    transferAmount
};

