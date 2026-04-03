// No database storage needed — all data is fetched live from Yahoo Finance via Python
export interface IStorage {}
export class NoopStorage implements IStorage {}
export const storage = new NoopStorage();
